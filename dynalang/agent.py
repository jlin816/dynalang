import sys

import embodied
import jax
import jax.numpy as jnp
import numpy as np
import ruamel.yaml as yaml
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
  def filter(self, record):
    return 'check_types' not in record.getMessage()
logger.addFilter(CheckTypesFilter())

from . import behaviors
from . import jaxagent
from . import jaxutils
from . import nets
from . import ninjax as nj
import optax


@jaxagent.Wrapper
class Agent(nj.Module):

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    with jax.transfer_guard("allow"):
      dummy_preproc = self.preprocess(
        {k: jnp.ones(v.shape) for k, v in self.obs_space.items()}) 
      preproc_shapes = {k: tuple(v.shape) for k, v in dummy_preproc.items() \
                        if not k.startswith("log_")}
    self.wm = WorldModel(obs_space, act_space, config, preproc_shapes, name='wm')
    self.preprocessors = {k: v() for k, v in
                          self.wm.encoder.preprocessors.items()}
    if self.config.run.pretrain_wm_only:
        print("Agent: Pretraining WM only.")
        return
    self.task_behavior = getattr(behaviors, config.task_behavior)(
        self.wm, self.act_space, self.config, name='task_behavior')
    if config.expl_behavior == 'None':
      self.expl_behavior = self.task_behavior
    else:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.wm, self.act_space, self.config, name='expl_behavior')

  def policy_initial(self, batch_size):
    return (
        self.wm.initial(batch_size),
        self.task_behavior.initial(batch_size),
        self.expl_behavior.initial(batch_size))

  def train_initial(self, batch_size):
    return self.wm.initial(batch_size)

  def policy(self, obs, state, mode='train'):
    self.config.jax.jit and print('Tracing policy function.')
    obs = self.preprocess(obs)
    (prev_latent, prev_action), task_state, expl_state = state
    embed = self.wm.encoder(obs)
    if self.config.rssm_type == "token":
      latent = self.wm.rssm.obs_step(
          prev_latent, prev_action, embed, obs["token"], obs['is_first'])
    else:
      latent = self.wm.rssm.obs_step(
          prev_latent, prev_action, embed, obs['is_first'])
    task_outs, task_state = self.task_behavior.policy(latent, task_state)
    expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)
    outs = {'eval': task_outs, 'explore': expl_outs, 'train': task_outs}[mode]
    state = ((latent, outs['action']), task_state, expl_state)
    return outs, state

  def train(self, data, state):
    self.config.jax.jit and print('Tracing train function.')
    metrics = {}
    data = self.preprocess(data)
    state, wm_outs, mets = self.wm.train(data, state)
    metrics.update(mets)
    context = {**data, **wm_outs['post']}
    if self.config.run.pretrain_wm_only:
        return wm_outs, state, metrics
    # Flatten (batch, seq) -> (batch * seq)
    start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
    _, mets = self.task_behavior.train(self.wm.imagine, start, context)
    metrics.update(mets)
    if self.config.expl_behavior != 'None':
      _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    outs = {}
    return outs, state, metrics

  def train_wm(self, data, state):
    metrics = {}
    data = self.preprocess(data)
    state, wm_outs, mets = self.wm.train(data, state)
    metrics.update(mets)
    context = {**data, **wm_outs['post']}
    return wm_outs, state, metrics

  def report(self, data):
    self.config.jax.jit and print('Tracing report function.')
    data = self.preprocess(data)
    report = {}
    report.update(self.wm.report(data))
    if self.config.run.pretrain_wm_only:
      return report
    mets = self.task_behavior.report(data)
    report.update({f'task_{k}': v for k, v in mets.items()})
    if self.expl_behavior is not self.task_behavior:
      mets = self.expl_behavior.report(data)
      report.update({f'expl_{k}': v for k, v in mets.items()})
    return report

  def vis(self, data, num_obs, num_imagine):
    data = self.preprocess(data)
    return self.wm.vis(data, num_obs, num_imagine)

  def save(self):
    data = jax.tree_util.tree_flatten(jax.tree_util.tree_map(
        jnp.asarray, self.state))[0]
    data = [np.asarray(x) for x in data]
    return data

  def load(self, state):
    self.state = jax.tree_util.tree_flatten(self.state)[1].unflatten(state)

  def preprocess(self, obs):
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if key == "token":
        value = jax.nn.one_hot(value, self.obs_space[key].high)
      elif len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      else:
        value = value.astype(jnp.float32)
      obs[key] = value
    obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
    return obs


class WorldModel(nj.Module):

  def __init__(self, obs_space, act_space, config, shapes):
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.config = config
#    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
#    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
    self.encoder = nets.MultiEncoder(shapes, **config.encoder, name='enc')
    if self.config.rssm_type == 'rssm':
      self.rssm = nets.RSSM(**config.rssm, name='rssm')
    elif self.config.rssm_type == 'early':
      self.rssm = nets.EarlyRSSM(**config.early_rssm, name='rssm')
    elif self.config.rssm_type == 'token':
      self.rssm = nets.TokenRSSM(**config.token_rssm, name='rssm')
    else:
      raise NotImplementedError(self.config.rssm_type)
    self.heads = {
        'decoder': nets.MultiDecoder(shapes, **config.decoder, name='dec'),
        'reward': nets.MLP((), **config.reward_head, name='rew'),
        'cont': nets.MLP((), **config.cont_head, name='cont')}
    self.opt = jaxutils.Optimizer(name='model_opt', **config.model_opt)
    scales = self.config.loss_scales.copy()
    image, vector = scales.pop('image'), scales.pop('vector')
    scales.update({k: image for k in self.heads['decoder'].cnn_shapes})
    scales.update({k: vector for k in self.heads['decoder'].mlp_shapes})
    self.scales = scales

  def initial(self, batch_size):
    prev_latent = self.rssm.initial(batch_size)
    prev_action = jnp.zeros((batch_size, *self.act_space.shape))
    return prev_latent, prev_action

  def train(self, data, state):
    for key in [x for x in self.config.zero_data_keys if x]:
      data[key] = jnp.zeros_like(data[key])
    modules = [self.encoder, self.rssm, *self.heads.values()]

    if self.config.skip_mlp_training:
      assert not self.config.skip_cnn_training
      enc = self.encoder._cnn
      dec = self.heads['decoder']._cnn
      others = {k: v for k, v in self.heads.items() if k != 'decoder'}
      modules = [self.rssm, enc, dec, *others.values()]
    if self.config.skip_cnn_training:
      assert not self.config.skip_mlp_training
      enc = self.encoder._mlp
      dec = self.heads['decoder']._mlp
      others = {k: v for k, v in self.heads.items() if k != 'decoder'}
      modules = [self.rssm, enc, dec, *others.values()]

    mets, (state, outs, metrics) = self.opt(
        modules, self.loss, data, state, has_aux=True)
    metrics.update(mets)
    return state, outs, metrics

  def loss(self, data, state):
    embed = self.encoder(
      data,
      zero_mlp=self.config.zero_mlp,
      zero_cnn=self.config.zero_cnn)
    prev_latent, prev_action = state
    prev_actions = jnp.concatenate([
        prev_action[:, None], data['action'][:, :-1]], 1)
    if self.config.rssm_type == "token":
      post = self.rssm.observe(
          prev_actions, embed, data["token"], data['is_first'], prev_latent)
    else:
      post = self.rssm.observe(
          embed, prev_actions, data['is_first'], prev_latent)
    dists = {}
    feats = {**post, 'embed': embed}
    for name, head in self.heads.items():
      inp = feats if name in self.config.grad_heads else sg(feats)
      out = head(inp)
      out = out if isinstance(out, dict) else {name: out}
      dists.update(out)
    losses = {}
    if self.config.rssm_type == "early":
      rssm_losses, prior = self.rssm.loss(post, prev_latent, prev_actions, **self.config.rssm_loss)
    elif self.config.rssm_type == "token":
      rssm_losses, prior = self.rssm.loss(
        post,
        prev_latent,
        prev_actions,
        data["token"],
        **self.config.rssm_loss
      )
    else:
      rssm_losses, prior = self.rssm.loss(post, **self.config.rssm_loss)

    # LM loss
    if self.scales["lm"] > 0:
      print("Adding LM loss")
      next_ac = data["action"][:, :-1].reshape((-1, 1, *data["action"].shape[2:]))
      context = {k: v[:, :-1].reshape((-1, *v.shape[2:]))
                 for k, v in post.items()}
      one_step_openl = self.heads["decoder"](
        self.rssm.imagine(next_ac, context),
      )
      truth = data["token"][:, 1:].reshape((-1, 1, *data["token"].shape[2:]))
      nll = -(one_step_openl["token"].log_prob(truth)).mean(-1)
#      nll = nll.reshape((data["token"].shape[0], -1)) # (batch, seq - 1)
      lm_loss = (nll * self.scales["lm"]).mean()
    else:
      lm_loss = 0

    losses.update(rssm_losses)
    for key, dist in dists.items():
      loss = -dist.log_prob(data[key].astype(jnp.float32))
      assert loss.shape == embed.shape[:2], (key, loss.shape)
      losses[key] = loss
    scaled = {k: v * self.scales[k] for k, v in losses.items()}
    model_loss = sum(scaled.values())
    out = {'embed':  embed, 'post': post, 'prior': prior}
    out.update({f'{k}_loss': v for k, v in losses.items()})
    last_latent = {k: v[:, -1] for k, v in post.items()}
    last_action = data['action'][:, -1]
    state = last_latent, last_action
    metrics = self._metrics(data, dists, post, prior, losses, model_loss)
    return model_loss.mean() + lm_loss, (state, out, metrics)

  def imagine(self, policy, start, horizon, carry=None):
    if carry is None:
      policy = lambda s, c, f=policy: (f(s), {})
      carry = {}
    state_keys = list(self.rssm.initial(1).keys())
    state = {k: v for k, v in start.items() if k in state_keys}
    action, carry = policy(state, carry)
    keys = list(state.keys()) + list(action.keys()) + list(carry.keys())
    assert len(set(keys)) == len(keys), ('Colliding keys', keys)
    def step(prev, _):
      state, action, carry = prev
      state = self.rssm.img_step(state, action['action'])
      action, carry = policy(state, carry)
      return state, action, carry
    states, actions, carries = jaxutils.scan(
        step, jnp.arange(horizon), (state, action, carry),
        self.config.imag_unroll)
    states, actions, carries = tree_map(
        lambda traj, first: jnp.concatenate([first[None], traj], 0),
        (states, actions, carries), (state, action, carry))
    traj = {**states, **actions, **carries}
    if self.config.imag_cont == 'mode':
      cont = self.heads['cont'](traj).mode()
    elif self.config.imag_cont == 'mean':
      cont = self.heads['cont'](traj).mean()
    else:
      raise NotImplementedError(self.config.imag_cont)
    first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
    traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
    discount = 1 - 1 / self.config.horizon
    traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
    return traj

  def report(self, data):
    # data: dict, each val with shape (batch, length, <obs shape>)
    state = self.initial(len(data['is_first']))
    report = {}
    report.update(self.loss(data, state)[-1][-1])
    if self.config.rssm_type == "token":
      context = self.rssm.observe(
          data['action'][:6, :5],
          self.encoder(data)[:6, :5],
          data['token'][:6, :5],
          data['is_first'][:6, :5])
    else:
      context = self.rssm.observe(
          self.encoder(data)[:6, :5], data['action'][:6, :5],
          data['is_first'][:6, :5])
    # context:
    # - deter (batch, prefix_len, rssm.deter)
    # - logit, stoch (batch, prefix_len, rssm.stoch, rssm.classes)
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    openl = self.heads['decoder'](
        self.rssm.imagine(data['action'][:6, 5:], start),
    )
    for key in self.heads['decoder'].cnn_shapes.keys():
      truth = data[key][:6].astype(jnp.float32)
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      error = (model - truth + 1) / 2
      video = jnp.concatenate([truth, model, error], 2)
      report[f'openl_{key}'] = jaxutils.video_grid(video)
    # 1 step prediction loss for text
    # Calculate text ppl over entire batch and seq len for more context
    # Above is buggy - observe takes in prev_actions, ac taken into this state (see loss)
    # data["action"] is action we took out of this state
    # prev_actions is action we took into this state
    if self.config.run.pretrain_wm_only and "token" in data: 
      prev_actions = jnp.concatenate([
        jnp.zeros_like(data["action"][:, 0:1]), # dummy first action
        data["action"][:, :-1]], 1)
      context = self.rssm.observe(
        self.encoder(data), prev_actions, data["is_first"])
      # a_t is action out of o_t, cut off last timestep since we don't have truth
      context = {k: v[:, :-1].reshape((-1, *v.shape[2:])) for k, v in context.items()}
      next_ac = data["action"][:, :-1].reshape((-1, 1, *data["action"].shape[2:]))
      one_step_openl = self.heads["decoder"](
        self.rssm.imagine(next_ac, context),
      )
      truth = data["token"][:, 1:].reshape((-1, 1, *data["token"].shape[2:]))
      nll = -(one_step_openl["token"].log_prob(truth)).mean(-1)
      ppl = jnp.exp(nll.mean())
      report["token_lm_nll_min"] = nll.min()
      report["token_lm_nll_max"] = nll.max()
      report["token_lm_nll_mean"] = nll.mean()
      report["token_lm_ppl"] = ppl
      # n-step openl text generation
      report["1step_openl_text"] = one_step_openl["token"].sample(seed=nj.rng()).argmax(-1) 
      report["nstep_openl_text"] = openl["token"].sample(seed=nj.rng()).argmax(-1)
    return report

  def vis(self, data, num_obs, num_imagine):
    assert data["action"].shape[0] == 1
    state = self.initial(len(data["is_first"]))
    prev_actions = jnp.concatenate([
      jnp.zeros_like(data["action"][:, 0:1]), # dummy first action
      data["action"][:, :-1]], 1)
    context = self.rssm.observe(
      embed=self.encoder(data)[:, :num_obs],
      action=prev_actions[:, :num_obs],#data["action"][:, :num_obs],
      is_first=data["is_first"][:, :num_obs])
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    end = num_obs + num_imagine
    openl = self.heads['decoder'](
      # note the last start state is index num_obs-1
      self.rssm.imagine(data['action'][:, num_obs-1:end], start),
    )
    reward = self.heads['reward'](
      self.rssm.imagine(data['action'][:, num_obs-1:end], start),
    )
    return recon, openl, reward
  
  def _metrics(self, data, dists, post, prior, losses, model_loss):
    entropy = lambda feat: self.rssm.get_dist(feat).entropy()
    metrics = {}
    metrics.update(jaxutils.tensorstats(entropy(prior), 'prior_ent'))
    metrics.update(jaxutils.tensorstats(entropy(post), 'post_ent'))
    metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    metrics['model_loss_mean'] = model_loss.mean()
    metrics['model_loss_std'] = model_loss.std()
    metrics['reward_max_data'] = jnp.abs(data['reward']).max()
    metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()
    if 'reward' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
      metrics.update({f'reward_{k}': v for k, v in stats.items()})
    if 'cont' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
      metrics.update({f'cont_{k}': v for k, v in stats.items()})
    return metrics


class ImagActorCritic(nj.Module):

  def __init__(self, critics, scales, act_space, config):
    critics = {k: v for k, v in critics.items() if scales[k]}
    for key, scale in scales.items():
      assert not scale or key in critics, key
    self.critics = {k: v for k, v in critics.items() if scales[k]}
    self.scales = scales
    self.act_space = act_space
    self.config = config
    disc = act_space.discrete
    self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
    self.actor = nets.MLP(
        name='actor', dims='deter', shape=act_space.shape, **config.actor,
        dist=config.actor_dist_disc if disc else config.actor_dist_cont)
    self.retnorms = {
        k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
        for k in critics}
    self.opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)

  def initial(self, batch_size):
    return {}

  def policy(self, state, carry, sample=True):
    dist = self.actor(sg(state))
    action = dist.sample(seed=nj.rng()) if sample else dist.mode()
    return {'action': action}, carry

  def train(self, imagine, start, context):
    carry = self.initial(len(start['deter']))
    def loss(start):
      traj = imagine(self.policy, start, self.config.imag_horizon, carry)
      loss, metrics = self.loss(traj)
      return loss, (traj, metrics)
    mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
    metrics.update(mets)
    for key, critic in self.critics.items():
      mets = critic.train(traj, self.actor)
      metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    return traj, metrics

  def loss(self, traj):
    metrics = {}
    advs = []
    total = sum(self.scales[k] for k in self.critics)
    for key, critic in self.critics.items():
      rew, ret, base = critic.score(traj, self.actor)
      offset, invscale = self.retnorms[key](ret)
      normed_ret = (ret - offset) / invscale
      normed_base = (base - offset) / invscale
      advs.append((normed_ret - normed_base) * self.scales[key] / total)
      metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
      metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
      metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
    adv = jnp.stack(advs).sum(0)
    policy = self.actor(sg(traj))
    logpi = policy.log_prob(sg(traj['action']))[:-1]
    loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]
    ent = policy.entropy()[:-1]
    loss -= self.config.actent * ent
    loss *= sg(traj['weight'])[:-1]
    loss *= self.config.loss_scales.actor
    metrics.update(self._metrics(traj, policy, logpi, ent, adv))
    return loss.mean(), metrics

  def _metrics(self, traj, policy, logpi, ent, adv):
    metrics = {}
    ent = policy.entropy()[:-1]
    rand = (ent - policy.minent) / (policy.maxent - policy.minent)
    rand = rand.mean(range(2, len(rand.shape)))
    act = traj['action']
    act = jnp.argmax(act, -1) if self.act_space.discrete else act
    metrics.update(jaxutils.tensorstats(act, 'action'))
    metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
    metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
    metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    return metrics


class VFunction(nj.Module):

  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), name='net', dims='deter', **self.config.critic)
    self.slow = nets.MLP((), name='slow', dims='deter', **self.config.critic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.opt = jaxutils.Optimizer(name='critic_opt', **self.config.critic_opt)

  def train(self, traj, actor):
    target = sg(self.score(traj, slow=self.config.slow_critic_target)[1])
    mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics

  def loss(self, traj, target):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(self.slow(traj).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    loss = (loss * sg(traj['weight'])).mean()
    loss *= self.config.loss_scales.critic
    metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, actor=None, slow=False):
    rew = self.rewfn(traj)
    assert len(rew) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    if slow:
      value = self.slow(traj).mean()
    else:
      value = self.net(traj).mean()
    vals = [value[-1]]
    interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    return rew, ret, value[:-1]
