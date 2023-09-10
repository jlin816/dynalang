import numpy as np


class Driver:

  def __init__(self, env, **kwargs):
    assert len(env) > 0
    self._env = env
    self._kwargs = kwargs
    self._callbacks = []
    self.reset()

  def reset(self):
    self._acts = {
        k: np.zeros((len(self._env),) + v.shape, v.dtype)
        for k, v in self._env.act_space.items()}
    self._acts['reset'] = np.ones(len(self._env), bool)
    self._state = None

  def on_step(self, callback):
    self._callbacks.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode):
    assert all(len(x) == len(self._env) for x in self._acts.values())
    acts = {k: v for k, v in self._acts.items() if not k.startswith('log_')}
    obs = self._env.step(acts)
    assert all(len(x) == len(self._env) for x in obs.values()), obs
    acts, self._state = policy(obs, self._state, **self._kwargs)
    if obs['is_last'].any():
      mask = ~obs['is_last']
      acts = {k: self._mask(v, mask) for k, v in acts.items()}
    acts['reset'] = obs['is_last'].copy()
    self._acts = acts
    trans = {**obs, **acts}
    for i in range(len(self._env)):
      trn = {k: v[i] for k, v in trans.items()}
      [fn(trn, i, **self._kwargs) for fn in self._callbacks]
    step += len(obs['is_first'])
    episode += obs['is_last'].sum()
    return step, episode

  def _mask(self, value, mask):
    while mask.ndim < value.ndim:
      mask = mask[..., None]
    return value * mask.astype(value.dtype)
