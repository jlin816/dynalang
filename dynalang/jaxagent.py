import concurrent.futures
import os

import embodied
import jax
import jax.numpy as jnp
import numpy as np

from . import jaxutils
from . import ninjax as nj

tree_map = jax.tree_util.tree_map
tree_flatten = jax.tree_util.tree_flatten
tree_leaves = jax.tree_util.tree_leaves


def Wrapper(agent_cls):
  class Agent(JAXAgent):
    configs = agent_cls.configs
    inner = agent_cls
    def __init__(self, *args, **kwargs):
      super().__init__(agent_cls, *args, **kwargs)
  return Agent


class JAXAgent(embodied.Agent):

  def __init__(self, agent_cls, obs_space, act_space, step, config):
    self.config = config.jax
    self.batch_size = config.batch_size
    self.batch_length = config.batch_length
    self.data_loaders = config.data_loaders
    self.logdir = embodied.Path(config.logdir)
    self._setup()
    self.agent = agent_cls(obs_space, act_space, step, config, name='agent')
    self.rng = np.random.default_rng(config.seed)

    available = jax.devices(self.config.platform)
    self.policy_devices = [available[i] for i in self.config.policy_devices]
    self.train_devices = [available[i] for i in self.config.train_devices]
    self.single_device = (self.policy_devices == self.train_devices) and (
        len(self.policy_devices) == 1)
    print(f'JAX devices ({jax.local_device_count()}):', available)
    print('Policy devices:', ', '.join([str(x) for x in self.policy_devices]))
    print('Train devices: ', ', '.join([str(x) for x in self.train_devices]))

    self._transform()
    self.varibs = self._init_varibs(obs_space, act_space)
    self.updates = embodied.Counter()
    self.once = True

    self.outs_worker = concurrent.futures.ThreadPoolExecutor(1)
    self.mets_worker = concurrent.futures.ThreadPoolExecutor(1)
    self.sync_worker = concurrent.futures.ThreadPoolExecutor(1)
    self.outs_promise = None
    self.mets_promise = None
    self.sync_promise = None

    self.should_sync = embodied.when.Every(self.config.sync_every)
    if not self.single_device:
      self.policy_varibs = self._copy_varibs(self.varibs)
    self.load_wm_ckpt = getattr(config, "load_wm_ckpt", False)
    if self.load_wm_ckpt:
        print("Only loading WM from agent ckpt.")

  def policy(self, obs, state=None, mode='train'):
    obs = {k: v for k, v in obs.items() if not k.startswith('log_')}
    obs = self.dataload_preprocess_single(obs)
    obs = self._convert_inps(obs, self.policy_devices)
    rng = self._next_rngs(self.policy_devices)
    varibs = self.varibs if self.single_device else self.policy_varibs

    if state is None:
      state, _ = self._init_policy(varibs, rng, obs['is_first'])
    else:
      state = tree_map(
          np.asarray, state, is_leaf=lambda x: isinstance(x, list))
      state = self._convert_inps(state, self.policy_devices)

    (outs, state), _ = self._policy(varibs, rng, obs, state, mode=mode)
    if not self.single_device:
      if self.sync_promise and self.sync_promise.done():
        self.policy_varibs = self.sync_promise.result()
        self.sync_promise = None
    outs = self._convert_outs(outs, self.policy_devices)

    # TODO: Consider keeping policy states in accelerator memory.
    state = self._convert_outs(state, self.policy_devices)

    return outs, state

  def train(self, data, state=None):
    data = data.copy()
    rng = data.pop('rng')
    if state is None:
      rng = self._next_rngs(self.train_devices)
      state, self.varibs = self._init_train(self.varibs, rng, data['is_first'])
    prev_varibs = self.varibs
    (outs, state, mets), self.varibs = self._train(
        self.varibs, rng, data, state)
    self.updates.increment()

    if not self.single_device:
      if not self.sync_promise and self.should_sync(self.updates):
        self.sync_promise = self.sync_worker.submit(
            self._copy_varibs, prev_varibs, block=True)

    return_outs = {}
    if self.outs_promise:
      return_outs = self.outs_promise.result()
    self.outs_promise = self.outs_worker.submit(
        self._convert_outs, outs, self.train_devices)

    return_mets = {}
    if self.mets_promise and self.mets_promise.done():
      return_mets = self.mets_promise.result()
      self.mets_promise = None
    if not self.mets_promise:
      # Only request metrics if we aren't currently waiting for previous
      # metrics. This means we'll skip the metrics of some training steps if
      # fetching them from device would slow down the training loop.
      self.mets_promise = self.mets_worker.submit(
          self._convert_mets, mets, self.train_devices)

    if self.once:
      self.once = False
      assert jaxutils.Optimizer.PARAM_COUNTS
      for name, count in jaxutils.Optimizer.PARAM_COUNTS.items():
        return_mets[f'params_{name}'] = float(count)

    if self.config.profiler:
      outdir, copyto = self.logdir, None
      if str(outdir).startswith(('gs://', '/gcs/')):
        copyto = outdir
        outdir = embodied.Path('/tmp/profiler')
        outdir.mkdirs()
      if self.updates == 100:
        print(f'Start JAX profiler ({str(outdir)})')
        jax.profiler.start_trace(str(outdir))
      if self.updates == 120:
        from embodied.core import path as pathlib
        print('Stop JAX profiler')
        jax.profiler.stop_trace()
        if copyto:
          pathlib.GFilePath(outdir).copy(copyto)
          print(f'Copied profiler result {outdir} to {copyto}')

    return return_outs, state, return_mets

  def train_wm(self, data, state=None):
    data = self._convert_inps(data)
    rng = self._next_rngs(mirror=not self.varibs)
    assert state is not None
    (outs, state, mets), self.varibs = self._train(
        self.varibs, rng, data, state)
    outs = self._convert_outs(outs)
    mets = self._convert_mets(mets)
    return outs, state, mets

  def report(self, data):
    # TODO: We could also do the same pipelining optimization used in train()
    # but it doesn't really matter because report() is not called as often.
    data = data.copy()
    rng = data.pop('rng')
    mets, _ = self._report(self.varibs, rng, data)
    mets = self._convert_mets(mets, self.train_devices)
    return mets

  def vis(self, data, num_obs, num_imagine):
    data = data.copy()
    rng = self._next_rngs(self.train_devices)
    (recon, openl, reward), _ = self._vis(self.varibs, rng, data, num_obs, num_imagine)
    return jax.device_get(recon), jax.device_get(openl), jax.device_get(reward)

  def dataset(self, generator):
    preprocessors = []
    if not isinstance(generator, list):
      generator = [generator] * self.batch_size
    batcher = embodied.Batcher(
        sources=generator,
        workers=self.data_loaders,
        postprocess=lambda x: {
            **self._convert_inps(x, self.train_devices),
            'rng': self._next_rngs(self.train_devices)},
        prefetch_source=4,
        prefetch_batch=1,
        preprocessors=self.agent.wm.encoder.preprocessors,
    )
    return batcher()

  def postprocess(self, batch):
    return {**self._convert_inps(batch, self.train_devices),
            'rng': self._next_rngs(self.train_devices)}

  def dataload_preprocess_single(self, obs):
    """Preprocessing for dataload-time, e.g. padding elements of a batch.

    Args:
        obs: A dict with a single unbatched observation from online training
    """
    if not self.agent.preprocessors or len(self.agent.preprocessors) == 0:
        return obs
    assert obs["reward"].shape == (1,)
    assert obs["language_info"].shape == (1,) and isinstance(obs["language_info"][0], str)
    obs_pp = {}
    for k, v in obs.items():
      if k in self.agent.preprocessors:
          # Add batch dimension: [v] = (batch=1, time=1)
        preproc = self.agent.preprocessors[k]([v])
        for preproc_key, preproc_val in preproc.items():
          # (batch=1, time=1, tok_seq_len) -> (time=1, tok_seq_len)
          obs_pp[f"{k}_{preproc_key}"] = preproc_val[0]
      else:
        obs_pp[k] = obs[k]
    return obs_pp

  def save(self):
    if len(self.train_devices) > 1:
      varibs = tree_map(lambda x: x[0], self.varibs)
    else:
      varibs = self.varibs
    varibs = jax.device_get(varibs)
    data = tree_map(np.asarray, varibs)
    return data

  def load(self, state):
    if self.load_wm_ckpt:
      # Remove replicated dim
      if len(self.train_devices) > 1:
        orig = tree_map(lambda x: x[0], self.varibs)
      else:
        orig = self.varibs
      # Load subset of vars from checkpoint, keep orig for the rest
      self.varibs = jaxutils.load_partial_checkpoint(
        orig, state, "agent/wm")
    else:
      expected = set(self.varibs)
      found = set(state)
      assert jax.tree_util.tree_structure(expected) == jax.tree_util.tree_structure(found), \
        (expected, found)
      self.varibs = state

    if len(self.train_devices) == 1:
      self.varibs = jax.device_put(self.varibs, self.train_devices[0])
    else:
      self.varibs = jax.device_put_replicated(self.varibs, self.train_devices)
    if not self.single_device:
      self.policy_varibs = self._copy_varibs(self.varibs)

  def _setup(self):
    try:
      import tensorflow as tf
      tf.config.set_visible_devices([], 'GPU')
      tf.config.set_visible_devices([], 'TPU')
    except Exception as e:
      print('Could not disable TensorFlow devices:', e)
    if not self.config.prealloc:
      os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
    xla_flags = []
    if self.config.logical_cpus:
      count = self.config.logical_cpus
      xla_flags.append(f'--xla_force_host_platform_device_count={count}')
    if xla_flags:
      os.environ['XLA_FLAGS'] = ' '.join(xla_flags)
    jax.config.update('jax_platform_name', self.config.platform)
    jax.config.update('jax_disable_jit', not self.config.jit)
    jax.config.update('jax_debug_nans', self.config.debug_nans)
    if self.config.transfer_guard:
      jax.config.update('jax_transfer_guard', 'disallow')
    if self.config.platform == 'cpu':
      jax.config.update('jax_disable_most_optimizations', self.config.debug)
    jaxutils.COMPUTE_DTYPE = getattr(jnp, self.config.precision)

  def _transform(self):
    self._init_policy = nj.pure(lambda x: self.agent.policy_initial(len(x)))
    self._init_train = nj.pure(lambda x: self.agent.train_initial(len(x)))
    self._policy = nj.pure(self.agent.policy)
    self._train = nj.pure(self.agent.train)
    self._report = nj.pure(self.agent.report)
    self._vis = nj.pure(self.agent.vis)
    if len(self.train_devices) == 1:
      kw = dict(device=self.train_devices[0])
      self._init_train = nj.jit(self._init_train, **kw)
      self._train = nj.jit(self._train, **kw)
      self._report = nj.jit(self._report, **kw)
    else:
      kw = dict(devices=self.train_devices)
      self._init_train = nj.pmap(self._init_train, 'i', **kw)
      self._train = nj.pmap(self._train, 'i', **kw)
      self._report = nj.pmap(self._report, 'i', **kw)
    if len(self.policy_devices) == 1:
      kw = dict(device=self.policy_devices[0])
      self._init_policy = nj.jit(self._init_policy, **kw)
      self._policy = nj.jit(self._policy, static=['mode'], **kw)
    else:
      kw = dict(devices=self.policy_devices)
      self._init_policy = nj.pmap(self._init_policy, 'i', **kw)
      self._policy = nj.pmap(self._policy, 'i', static=['mode'], **kw)

  def _convert_inps(self, value, devices, rng=False, block=False):
    if len(devices) == 1:
      value = jax.device_put(value, devices[0])
    else:
      check = tree_map(lambda x: len(x) % len(devices) == 0, value)
      if not all(jax.tree_util.tree_leaves(check)):
        shapes = tree_map(lambda x: x.shape, value)
        raise ValueError(
            f'Batch must by divisible by {len(devices)} devices: {shapes}')
      value = tree_map(
          lambda x: x.reshape((len(devices), -1) + x.shape[1:]), value)
      shards = []
      for i in range(len(devices)):
        shards.append(tree_map(lambda x: x[i], value))
      value = jax.device_put_sharded(shards, devices)
    if rng:
      value['rng'] = self._next_rngs(devices)
    if block:
      jax.block_until_ready(value)
    return value

  def _convert_outs(self, value, devices):
    value = jax.device_get(value)
    if len(devices) > 1:
      value = tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), value)
    return value

  def _convert_mets(self, value, devices):
    if len(devices) > 1:
      value = tree_map(lambda x: x[0], value)
    return jax.device_get(value)

  def _next_rngs(self, devices, mirror=False, high=2 ** 63 - 1):
    if len(devices) == 1:
      return jax.device_put(self.rng.integers(high), devices[0])
    elif mirror:
      return jax.device_put_replicated(
          self.rng.integers(high), devices)
    else:
      return jax.device_put_sharded(
          list(self.rng.integers(high, size=len(devices))), devices)

  def _init_varibs(self, obs_space, act_space):
    varibs = {}
    rng = self._next_rngs(self.train_devices, mirror=True)
    dims = (self.batch_size, self.batch_length)
    data = self._dummy_batch({**obs_space, **act_space}, dims)
    data = self._convert_inps(data, self.train_devices)
    state, varibs = self._init_train(varibs, rng, data['is_first'])
    varibs = self._train(varibs, rng, data, state, init_only=True)
    # obs = self._dummy_batch(obs_space, (1,))
    # state, varibs = self._init_policy(varibs, rng, obs['is_first'])
    # varibs = self._policy(
    #     varibs, rng, obs, state, mode='train', init_only=True)
    return varibs

  def _copy_varibs(self, varibs, block=False):
    if self.single_device:
      return varibs
    if len(self.train_devices) > 1:
      varibs = tree_map(lambda x: x[0].device_buffer, self.varibs)
    if len(self.policy_devices) == 1:
      varibs = jax.device_put(varibs, self.policy_devices[0])
    else:
      varibs = jax.device_put_replicated(varibs, self.policy_devices)
    if block:
      jax.block_until_ready(varibs)
    return varibs

  def _dummy_batch(self, spaces, batch_dims):
    spaces = [(k, v) for k, v in spaces.items() if not k.startswith('log_')]
    data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces}
    for dim in reversed(batch_dims):
      data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
    return data
