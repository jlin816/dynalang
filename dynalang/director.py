from functools import partial as bind

import embodied
import jax
import jax.numpy as jnp
import numpy as np
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
f32 = jnp.float32

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from . import agent
from . import jaxutils
from . import nets
from . import ninjax as nj


"""
TODO: changes to original impl, we didn't apply these:
- goal autoencoder free nats instead of target kl
- manager actent: 0.5
- different return normalization
- small config changes
"""


class Director(nj.Module):

  def __init__(self, wm, act_space, config):
    self.wm = wm
    self.config = config
    self.extr_reward = lambda s: wm.heads['reward'](s).mean()[1:]
    self.act_space = act_space
    VF = agent.VFunction

    wconfig = config.update({
        'actor.inputs': self.config.worker_inputs,
        'critic.inputs': self.config.worker_inputs,
        'actent': self.config.worker_actent,
    })
    self.worker = agent.ImagActorCritic({
        'extr': VF(lambda s: s['reward_extr'], wconfig, name='w_extr_v'),
        'expl': VF(lambda s: s['reward_expl'], wconfig, name='w_expl_v'),
        'goal': VF(lambda s: s['reward_goal'], wconfig, name='w_goal_v'),
    }, config.worker_rews, act_space, wconfig, name='worker')

    mgr_act_space = {'skill': embodied.Space(np.int32, config.skill_shape)}
    mconfig = config.update({
        'actent': self.config.manager_actent,
    })
    self.manager = agent.ImagActorCritic({
        'extr': VF(lambda s: s['reward_extr'], mconfig, name='m_extr_v'),
        'expl': VF(lambda s: s['reward_expl'], mconfig, name='m_expl_v'),
        'goal': VF(lambda s: s['reward_goal'], mconfig, name='m_goal_v'),
    }, config.manager_rews, mgr_act_space, mconfig, name='manager')

    self.goal_shape = (config.rssm.deter,)
    self.goal_feat = nets.Input(['deter'])
    self.enc = nets.MLP(
        config.skill_shape, **config.goal_enc, dims='deter', name='enc')
    self.dec = nets.MLP(
        self.goal_shape, **config.goal_dec, dims='deter', name='dec')
    self.opt = jaxutils.Optimizer(name='goal_opt', **config.goal_opt)
    logits = jax.device_put(np.zeros(config.skill_shape))
    self.prior = tfd.Independent(
        jaxutils.OneHotDist(logits),
        len(config.skill_shape) - 1)

  def initial(self, batch_size):
    return {
        'step': jnp.zeros((batch_size,), jnp.int64),
        'skill': jnp.zeros((batch_size,) + self.config.skill_shape, f32),
        'goal': jnp.zeros((batch_size,) + self.goal_shape, f32),
    }

  def policy(self, latent, carry, imag=False):
    duration = self.config.train_skill_duration if imag else (
        self.config.env_skill_duration)
    skill = sg(jaxutils.switch(
        carry['step'] % duration == 0,
        self.manager.actor(latent)['skill'].sample(seed=nj.rng()),
        carry['skill']))
    goal = sg(jaxutils.switch(
        carry['step'] % duration == 0,
        self.dec({**latent, 'skill': skill}).mode(),
        carry['goal']))
    dist = self.worker.actor(sg({**latent, 'goal': goal}))
    outs = {k: v.sample(seed=nj.rng()) for k, v in dist.items()}
    # TODO: Visualization
    # if 'image' in self.wm.heads['decoder'].shapes:
    #   outs['log_goal'] = self.wm.heads['decoder']({
    #       'deter': goal, 'stoch': self.wm.rssm.get_stoch(goal),
    #   })['image'].mode()
    carry = {'step': carry['step'] + 1, 'skill': skill, 'goal': goal}
    return outs, carry

  def train(self, imagine, start, data):
    metrics = {}
    metrics.update(self.train_vae(data))
    if self.config.director_jointly:
      metrics.update(self.train_jointly(imagine, start))
    else:
      raise NotImplementedError
      # for impl in self.config.worker_goals:
      #   goal = self.propose_goal(start, impl)
      #   metrics.update(self.train_worker(imagine, start, goal)[1])
      # metrics.update(self.train_manager(imagine, start)[1])
    return None, metrics

  # def train_jointly(self, imagine, start):
  #   start = start.copy()
  #   metrics = {}
  #   with tf.GradientTape(persistent=True) as tape:
  #     policy = functools.partial(self.policy, imag=True)
  #     traj = self.wm.imagine_carry(
  #         policy, start, self.config.imag_horizon,
  #         self.initial(len(start['is_first'])))
  #     traj['reward_extr'] = self.extr_reward(traj)
  #     traj['reward_expl'] = self.expl_reward(traj)
  #     traj['reward_goal'] = self.goal_reward(traj)
  #     wtraj = self.split_traj(traj)
  #     mtraj = self.abstract_traj(traj)
  #   mets = self.worker.update(wtraj, tape)
  #   metrics.update({f'worker_{k}': v for k, v in mets.items()})
  #   mets = self.manager.update(mtraj, tape)
  #   metrics.update({f'manager_{k}': v for k, v in mets.items()})
  #   return traj, metrics

  def train_jointly(self, imagine, start):
    start = start.copy()
    metrics = {}

    def wloss(start):
      traj = imagine(
          bind(self.policy, imag=True), start, self.config.imag_horizon,
          carry=self.initial(len(start['is_first'])))
      scales = self.config.reward_scales
      traj['reward_extr'] = self.extr_reward(traj) * scales.extr
      traj['reward_expl'] = self.expl_reward(traj) * scales.expl
      traj['reward_goal'] = self.goal_reward(traj) * scales.goal
      wtraj = self.split_traj(traj)
      mtraj = self.abstract_traj(traj)
      wloss, wmets = self.worker.loss(wtraj)
      return wloss, (wtraj, mtraj, wmets)

    mets, (wtraj, mtraj, wmets) = self.worker.opt(
        self.worker.actor, wloss, start, has_aux=True)
    wmets.update(mets)
    for key, critic in self.worker.critics.items():
      mets = critic.train(wtraj, self.worker.actor)
      wmets.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    metrics.update({f'worker_{k}': v for k, v in wmets.items()})

    mets, mmets = self.manager.opt(
        self.manager.actor, self.manager.loss, mtraj, has_aux=True)
    mmets.update(mets)
    for key, critic in self.manager.critics.items():
      mets = critic.train(mtraj, self.manager.actor)
      mmets.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    metrics.update({f'manager_{k}': v for k, v in mmets.items()})

    return metrics

  # def train_manager(self, imagine, start):
  #   start = start.copy()
  #   with jnp.GradientTape(persistent=True) as tape:
  #     policy = functools.partial(self.policy, imag=True)
  #     traj = self.wm.imagine_carry(
  #         policy, start, self.config.imag_horizon,
  #         self.initial(len(start['is_first'])))
  #     traj['reward_extr'] = self.extr_reward(traj)
  #     traj['reward_expl'] = self.expl_reward(traj)
  #     traj['reward_goal'] = self.goal_reward(traj)
  #     mtraj = self.abstract_traj(traj)
  #   metrics = self.manager.update(mtraj, tape)
  #   metrics = {f'manager_{k}': v for k, v in metrics.items()}
  #   return traj, metrics

  # def train_worker(self, imagine, start, goal):
  #   start = start.copy()
  #   metrics = {}
  #   sg = lambda x: jnp.nest.map_structure(sg, x)
  #   with jnp.GradientTape(persistent=True) as tape:
  #     worker = lambda s: self.worker.actor(sg({**s, 'goal': goal})).sample(seed=nj.rng())
  #     traj = imagine(worker, start, self.config.imag_horizon)
  #     traj['goal'] = jnp.repeat(goal[None], 1 + self.config.imag_horizon, 0)
  #     traj['reward_extr'] = self.extr_reward(traj)
  #     traj['reward_expl'] = self.expl_reward(traj)
  #     traj['reward_goal'] = self.goal_reward(traj)
  #   mets = self.worker.update(traj, tape)
  #   metrics.update({f'worker_{k}': v for k, v in mets.items()})
  #   return traj, metrics

  def train_vae(self, data):
    def loss(data):
      metrics = {}
      goal = self.goal_feat(data).astype(f32)
      enc = self.enc({**data, 'goal': goal})
      dec = self.dec({**data, 'skill': enc.sample(seed=nj.rng())})
      rec = -dec.log_prob(sg(goal))
      kl = tfd.kl_divergence(enc, self.prior)
      kl = jnp.maximum(self.config.goal_kl_free, kl)
      assert rec.shape == kl.shape, (rec.shape, kl.shape)
      metrics['goalkl_mean'] = kl.mean()
      metrics['goalkl_std'] = kl.std()
      metrics['goalrec_mean'] = rec.mean()
      metrics['goalrec_std'] = rec.std()
      loss = (rec + self.config.goal_kl_scale * kl).mean()
      return loss, metrics
    metrics, mets = self.opt([self.enc, self.dec], loss, data, has_aux=True)
    metrics.update(mets)
    return metrics

  def propose_goal(self, start, impl):
    if impl == 'replay':
      feat = self.goal_feat(start).astype(f32)
      target = jax.random.permutation(nj.rng(), feat).astype(f32)
      skill = self.enc({**start, 'goal': target}).sample(seed=nj.rng())
      return self.dec({**start, 'skill': skill}).mode()
    if impl == 'replay_direct':
      feat = self.goal_feat(start).astype(f32)
      return jax.random.permutation(nj.rng(), feat).astype(f32)
    if impl == 'manager':
      skill = self.manager.actor(start)['skill'].sample(seed=nj.rng())
      return self.dec({**start, 'skill': skill}).mode()
    if impl == 'prior':
      skill = self.prior.sample(len(start['is_terminal']), seed=nj.rng())
      return self.dec({**start, 'skill': skill}).mode()
    raise NotImplementedError(impl)

  def goal_reward(self, traj):
    feat = self.goal_feat(traj).astype(f32)
    goal = sg(traj['goal'].astype(f32))
    gnorm = jnp.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
    fnorm = jnp.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
    norm = jnp.maximum(gnorm, fnorm)
    return jnp.einsum('...i,...i->...', goal / norm, feat / norm)[1:]

  def expl_reward(self, traj):
    feat = self.goal_feat(traj).astype(f32)
    enc = self.enc({**traj, 'goal': feat})
    dec = self.dec({**traj, 'skill': enc.sample(seed=nj.rng())})
    return ((dec.mode() - feat) ** 2).mean(-1)[1:]

  def split_traj(self, traj):
    traj = traj.copy()
    k = self.config.train_skill_duration
    for key, x in list(traj.items()):
      if key.startswith('reward_'):
        x = jnp.concatenate([0 * x[:1], x], 0)
      # T x B x F... -> B' x T' x B x F...
      x = x.reshape((x.shape[0] // k, k) + x.shape[1:])
      # B' x T' x B x F... -> T' x (B' B) x F...
      x = x.transpose((1, 0) + tuple(range(2, len(x.shape))))
      x = x.reshape((x.shape[0], -1, *x.shape[3:]))
      if key.startswith('reward_'):
        x = x[1:]
      traj[key] = x
    return traj

  def abstract_traj(self, traj):
    traj = traj.copy()
    # traj['action'] = traj.pop('skill')
    k = self.config.train_skill_duration
    reshape = lambda x: x.reshape((x.shape[0] // k, k, *x.shape[1:]))
    w = jnp.cumprod(reshape(traj['cont']), 1)
    for key, x in list(traj.items()):
      if key.startswith('reward_'):
        x = (reshape(jnp.concatenate([0 * x[:1], x], 0)) * w).mean(1)[1:]
      elif key == 'cont':
        x = reshape(x).prod(1)
      elif key == 'weights':
        x = reshape(x)[:, 0]
      else:
        x = reshape(x)[:, 0]
      traj[key] = x
    return traj

  def report(self, data):
    metrics = {}
    for impl in ('manager', 'prior', 'replay'):
      for key, video in self.report_worker(data, impl).items():
        metrics[f'impl_{impl}_{key}'] = video
    return metrics

  def report_worker(self, data, impl):
    # Prepare initial state.
    decoder = self.wm.heads['decoder']
    states = self.wm.rssm.observe(
        self.wm.encoder(data)[:6],
        {k: data[k][:6] for k in self.act_space},
        data['is_first'][:6],
        self.wm.rssm.initial(len(data['is_first'][:6])))
    start = {k: v[:, 4] for k, v in states.items()}
    start['is_terminal'] = data['is_terminal'][:6, 4]
    goal = self.propose_goal(start, impl)
    # Worker rollout.
    traj = self.wm.imagine(
        lambda s, c: self.worker.policy({**s, 'goal': goal}, c),
        start, self.config.worker_report_horizon, {})
    # Decode into images.
    initial = decoder(start)
    stoch = self.wm.rssm._prior(goal, sample=True)['stoch']
    target = decoder({'deter': goal, 'stoch': stoch})
    rollout = decoder(traj)
    # Stich together into videos.
    videos = {}
    for k in rollout.keys():
      if k not in decoder.cnn_keys:
        continue
      length = 1 + self.config.worker_report_horizon
      rows = []
      rows.append(jnp.repeat(initial[k].mode()[:, None], length, 1))
      if target is not None:
        rows.append(jnp.repeat(target[k].mode()[:, None], length, 1))
      rows.append(rollout[k].mode().transpose((1, 0, 2, 3, 4)))
      videos[k] = jaxutils.video_grid(jnp.concatenate(rows, 2))
    return videos
