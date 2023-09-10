import functools
import re

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
f32 = jnp.float32
tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

from . import jaxutils
from . import ninjax as nj
cast = jaxutils.cast_to_compute

from typing import List

class RSSM(nj.Module):

  def __init__(
      self, impl='softmax', deter=1024, stoch=32, classes=32, unroll=False,
      unimix=0.01, action_clip=1.0, bottleneck=-1, maskgit={}, **kw):
    assert impl in ('gaussian', 'softmax', 'maskgit'), impl
    self._impl = impl
    self._deter = deter
    self._stoch = stoch
    self._classes = classes
    self._unroll = unroll
    self._unimix = unimix
    self._action_clip = action_clip
    self._bottleneck = bottleneck
    self._kw = kw
    if self._impl == 'maskgit':
      from . import maskgit as mg
      self._maskgit = mg.MaskGit(stoch, classes, **maskgit, name='maskgit')

  def initial(self, batch_size):
    if self._impl == 'gaussian':
      state = dict(
          deter=jnp.zeros([batch_size, self._deter], f32),
          mean=jnp.zeros([batch_size, self._stoch], f32),
          std=jnp.ones([batch_size, self._stoch], f32),
          stoch=jnp.zeros([batch_size, self._stoch], f32))
    if self._impl == 'softmax':
      state = dict(
          deter=jnp.zeros([batch_size, self._deter], f32),
          logit=jnp.zeros([batch_size, self._stoch, self._classes], f32),
          stoch=jnp.zeros([batch_size, self._stoch, self._classes], f32))
    if self._impl == 'maskgit':
      state = dict(
          deter=jnp.zeros([batch_size, self._deter], f32),
          logit=jnp.zeros([batch_size, self._stoch, self._classes], f32),
          stoch=jnp.zeros([batch_size, self._stoch, self._classes], f32),
          mask=jnp.zeros([batch_size, self._stoch], bool))
    deter = self.get('initial', jnp.zeros, state['deter'][0].shape, f32)
    state['deter'] = jnp.repeat(jnp.tanh(deter)[None], batch_size, 0)
    state['stoch'] = self._prior(cast(state['deter']), sample=True)['stoch']
    return cast(state)

  def observe(self, embed, action, is_first, state=None):
    state = state or self.initial(action.shape[0])
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    step = lambda prev, inputs: self.obs_step(prev, *inputs)
    inputs = swap(action), swap(embed), swap(is_first)
    post = jaxutils.scan(step, inputs, state, self._unroll)
    post = {k: swap(v) for k, v in post.items()}
    return post

  def imagine(self, action, state=None):
    state = state or self.initial(action.shape[0])
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    action = swap(action)
    prior = jaxutils.scan(self.img_step, action, state, self._unroll)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def obs_step(self, prev_state, prev_action, embed, is_first):
    deter = self._gru(prev_state, prev_action, is_first)
    x = jnp.concatenate([deter, embed], -1)
    x = self.get('obs_out', Linear, **self._kw)(x)
    stats = self._stats('obs_stats', x)
    stoch = self.get_dist(stats).sample(seed=nj.rng())
    post = {'deter': deter, 'stoch': stoch, **stats}
    return cast(post)

  def img_step(self, prev_state, prev_action):
    deter = self._gru(prev_state, prev_action)
    return self._prior(deter, sample=True)

  def get_dist(self, stats):
    if self._impl == 'gaussian':
      mean = stats['mean'].astype(f32)
      std = stats['std'].astype(f32)
      return tfd.Independent(tfd.Normal(mean, std), 1)
    if self._impl == 'softmax':
      logit = stats['logit'].astype(f32)
      return tfd.Independent(jaxutils.OneHotDist(logit), 1)
    if self._impl == 'maskgit':
      logit = stats['logit'].astype(f32)
      return jaxutils.OneHotDist(logit)

  def loss(self, post, free=1.0):
    prior = self._prior(post['deter'], sample=False, post=post)
    if self._impl == 'gaussian':
      dyn = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
      rep = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
    if self._impl == 'softmax':
      dyn = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
      rep = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
    if self._impl == 'maskgit':
      dyn = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
      rep = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
      dyn = (dyn * prior['mask']).sum(-1) / prior['mask'].sum(-1)
      rep = (rep * prior['mask']).sum(-1) / prior['mask'].sum(-1)
    if free:
      dyn = jnp.maximum(dyn, free)
      rep = jnp.maximum(rep, free)
    return {'dyn': dyn, 'rep': rep}, prior

  def _prior(self, deter, sample, post=None):
    if self._impl == 'gaussian':
      x = self.get('img_out', Linear, **self._kw)(deter)
      stats = self._stats('img_stats', x)
      stoch = self.get_dist(stats).sample(seed=nj.rng()) if sample else None
      return cast({'deter': deter, 'stoch': stoch, **stats})
    if self._impl == 'softmax':
      x = self.get('img_out', Linear, **self._kw)(deter)
      stats = self._stats('img_stats', x)
      stoch = self.get_dist(stats).sample(seed=nj.rng()) if sample else None
      return cast({'deter': deter, 'stoch': stoch, **stats})
    if self._impl == 'maskgit':
      if sample:
        logit = jnp.zeros((*deter.shape[:-1], self._stoch, self._classes), f32)
        mask = jnp.ones((*deter.shape[:-1], self._stoch), bool)
        stats = {'logit': logit, 'mask': mask}
        stoch = self._maskgit.sample(deter.reshape(((-1, deter.shape[-1]))))
        stoch = stoch.reshape((*deter.shape[:-1], *stoch.shape[1:]))
      else:
        assert post is not None
        stoch = post['stoch']
        logit, mask = self._maskgit.train(
            stoch.reshape((-1, *stoch.shape[-2:])),
            deter.reshape((-1, *deter.shape[-1:])))
        logit = logit.reshape((*deter.shape[:-1], *logit.shape[1:]))
        mask = mask.reshape((*deter.shape[:-1], *mask.shape[1:]))
        stats = {'logit': logit, 'mask': mask}
        stoch = None
      return cast({'stoch': stoch, 'deter': deter, **stats})

  def _gru(self, prev_state, prev_action, is_first=None):
    prev_action = cast(prev_action)
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
    if is_first is not None:
      prev_state, prev_action = tree_map(
          lambda prev, init: jaxutils.switch(is_first, init, prev),
          (prev_state, prev_action),
          (self.initial(len(is_first)), jnp.zeros_like(prev_action)))
    batch_shape = prev_state['deter'].shape[:-1]
    x = jnp.concatenate([
        prev_state['stoch'].reshape((*batch_shape, -1)),
        cast(prev_action).reshape((*batch_shape, -1))], -1)
    x = self.get('img_in', Linear, **self._kw)(x)
    x = jnp.concatenate([prev_state['deter'], x], -1)
    if self._bottleneck > 0:
      kw = {**self._kw, 'units': self._bottleneck}
      x = self.get('bottleneck', Linear, **kw)(x)
    kw = {**self._kw, 'act': 'none', 'units': 3 * self._deter}
    x = self.get('gru', Linear, **kw)(x)
    reset, cand, update = jnp.split(x, 3, -1)
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * prev_state['deter']
    return deter

  def _stats(self, name, x):
    if self._impl == 'gaussian':
      x = self.get(name, Linear, 2 * self._stoch)(x)
      mean, std = jnp.split(x, 2, -1)
      std = 2 * jax.nn.sigmoid(std / 2) + 0.1
      return {'mean': mean, 'std': std}
    if self._impl == 'softmax':
      x = self.get(name, Linear, self._stoch * self._classes)(x)
      logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
      if self._unimix:
        probs = jax.nn.softmax(logit, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        logit = jnp.log(probs)
      return {'logit': logit}
    if self._impl == 'maskgit':
      x = self.get(name, Linear, self._stoch * self._classes)(x)
      logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
      if self._unimix:
        probs = jax.nn.softmax(logit, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        logit = jnp.log(probs)
      mask = jnp.ones((x.shape[0], self._stoch), bool)
      return {'logit': logit, 'mask': mask}

class TokenRSSM(nj.Module):

  def __init__(
      self, deter=1024, stoch=32, classes=32, vocab=256, unroll=False,
      unimix=0.01, action_clip=1.0, bottleneck=-1, prior_layers=3, **kw):
    self._deter = deter
    self._stoch = stoch
    self._classes = classes
    self._vocab = vocab
    self._unroll = unroll
    self._unimix = unimix
    self._action_clip = action_clip
    self._bottleneck = bottleneck
    self._prior_layers = prior_layers
    self._kw = kw

  def initial(self, batch_size):
    deter = self.get('initial', jnp.zeros, [self._deter], f32)
    state = dict(
        deter=jnp.repeat(jnp.tanh(deter)[None], batch_size, 0),
        z_logit=jnp.zeros([batch_size, self._stoch, self._classes], f32),
        z_stoch=jnp.zeros([batch_size, self._stoch, self._classes], f32),
        l_logit=jnp.zeros([batch_size, self._vocab], f32),
        l_stoch=jnp.zeros([batch_size, self._vocab], f32))
    return cast(state)

  def observe(self, action, embed, token, is_first, state=None):
    state = state or self.initial(action.shape[0])
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    step = lambda prev, inputs: self.obs_step(prev, *inputs)
    inputs = tree_map(swap, (action, embed, token, is_first))
    post = jaxutils.scan(step, inputs, state, self._unroll)
    post = {k: swap(v) for k, v in post.items()}
    return post

  def imagine(self, action, state=None):
    state = state or self.initial(action.shape[0])
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    action = swap(action)
    prior = jaxutils.scan(self.img_step, action, state, self._unroll)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def obs_step(self, prev_state, prev_action, embed, token, is_first):
    prev_state, prev_action = tree_map(
        lambda prev, init: jaxutils.switch(is_first, init, prev),
        (prev_state, prev_action),
        (self.initial(len(is_first)), jnp.zeros_like(prev_action)))
    rep = self._repr(embed, token, sample=True)
    inp = self._inps(rep['z_stoch'], rep['l_stoch'], prev_action)
    deter = self._core(prev_state['deter'], inp)
    return cast({**rep, 'deter': deter})

  def img_step(self, prev_state, prev_action):
    pred = self._pred(prev_state['deter'], prev_action, sample=True)
    inp = self._inps(pred['z_stoch'], pred['l_stoch'], prev_action)
    deter = self._core(prev_state['deter'], inp)
    return cast({**pred, 'deter': deter})

  def get_dist(self, stats):  # For backwards compatibility.
    return self.get_dist_z(stats)

  def get_dist_z(self, stats):
    return tfd.Independent(jaxutils.OneHotDist(
        stats['z_logit'].astype(f32)), 1)

  def get_dist_l(self, stats):
    return jaxutils.OneHotDist(stats['l_logit'].astype(f32))

  def loss(self, post, prev_state, prev_action, token, free=1.0):
    prev_deter = jnp.concatenate([
        prev_state['deter'][:, None], post['deter'][:, :-1]], 1)
    pred = self._pred(prev_deter, prev_action, sample=False)
    dyn = self.get_dist_z(sg(post)).kl_divergence(self.get_dist_z(pred))
    rep = self.get_dist_z(post).kl_divergence(self.get_dist_z(sg(pred)))
    if free:
      dyn = jnp.maximum(dyn, free)
      rep = jnp.maximum(rep, free)
    token = -self.get_dist_l(post).log_prob(token)
    losses = {'dyn': dyn, 'rep': rep, 'token': token}
    return losses, pred

  def _inps(self, z_stoch, l_stoch, act):
    batch_shape = z_stoch.shape[:-2]
    if self._action_clip > 0.0:
      act *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(act)))
    x = jnp.concatenate([
        cast(act).reshape((*batch_shape, -1)),
        cast(z_stoch).reshape((*batch_shape, -1)),
        cast(l_stoch).reshape((*batch_shape, -1))], -1)
    x = self.get('inps', Linear, **self._kw)(x)
    return x

  def _repr(self, embed, token, sample=True):
    x = self.get('repr', Linear, **self._kw)(embed)
    z_logit = self.get('repr_z', Linear, (self._stoch, self._classes))(x)
    z_logit = self._apply_unimix(z_logit)
    l_logit = jnp.zeros_like(token)
    rep = {'z_logit': z_logit, 'l_logit': l_logit}
    if sample:
      rep['z_stoch'] = self.get_dist_z(rep).sample(seed=nj.rng())
      rep['l_stoch'] = token
    return cast(rep)

  def _pred(self, deter, act, sample=True):
    x = jnp.concatenate([deter, act], -1)
    for i in range(self._prior_layers):
      x = self.get(f'pred{i}', Linear, **self._kw)(x)
    z_logit = self.get('pred_z', Linear, (self._stoch, self._classes))(x)
    z_logit = self._apply_unimix(z_logit)
    l_logit = self.get('pred_l', Linear, self._vocab)(x)
    pred = {'z_logit': z_logit, 'l_logit': l_logit}
    if sample:
      pred['z_stoch'] = self.get_dist_z(pred).sample(seed=nj.rng())
      pred['l_stoch'] = self.get_dist_l(pred).sample(seed=nj.rng())
    return cast(pred)

  def _core(self, prev_deter, inputs):
    x = jnp.concatenate([prev_deter, inputs], -1)
    if self._bottleneck > 0:
      kw = {**self._kw, 'units': self._bottleneck}
      x = self.get('bottleneck', Linear, **kw)(x)
    kw = {**self._kw, 'act': 'none', 'units': 3 * self._deter}
    x = self.get('gru', Linear, **kw)(x)
    reset, cand, update = jnp.split(x, 3, -1)
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * prev_deter
    return deter

  def _apply_unimix(self, logit):
    if not self._unimix:
      return logit
    probs = jax.nn.softmax(logit, -1)
    uniform = jnp.ones_like(probs) / probs.shape[-1]
    probs = (1 - self._unimix) * probs + self._unimix * uniform
    logit = jnp.log(probs)
    return logit


class EarlyRSSM(nj.Module):

  def __init__(
      self, deter=1024, stoch=32, classes=32, unroll=False,
      unimix=0.01, action_clip=1.0, bottleneck=-1, prior_layers=3, **kw):
    self._deter = deter
    self._stoch = stoch
    self._classes = classes
    self._unroll = unroll
    self._unimix = unimix
    self._action_clip = action_clip
    self._bottleneck = bottleneck
    self._prior_layers = prior_layers
    self._kw = kw

  def initial(self, batch_size):
    deter = self.get('initial', jnp.zeros, [self._deter], f32)
    state = dict(
        deter=jnp.repeat(jnp.tanh(deter)[None], batch_size, 0),
        logit=jnp.zeros([batch_size, self._stoch, self._classes], f32),
        stoch=jnp.zeros([batch_size, self._stoch, self._classes], f32))
    return cast(state)

  def observe(self, embed, action, is_first, state=None):
    state = state or self.initial(action.shape[0])
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    step = lambda prev, inputs: self.obs_step(prev, *inputs)
    inputs = swap(action), swap(embed), swap(is_first)
    post = jaxutils.scan(step, inputs, state, self._unroll)
    post = {k: swap(v) for k, v in post.items()}
    return post

  def imagine(self, action, state=None):
    state = state or self.initial(action.shape[0])
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    action = swap(action)
    prior = jaxutils.scan(self.img_step, action, state, self._unroll)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def obs_step(self, prev_state, prev_action, embed, is_first):
    prev_state, prev_action = tree_map(
        lambda prev, init: jaxutils.switch(is_first, init, prev),
        (prev_state, prev_action),
        (self.initial(len(is_first)), jnp.zeros_like(prev_action)))
    rep = self._repr(embed, sample=True)
    inp = self._inps(rep['stoch'], prev_action)
    deter = self._core(prev_state['deter'], inp)
    return cast({**rep, 'deter': deter})

  def img_step(self, prev_state, prev_action):
    pred = self._pred(prev_state['deter'], prev_action, sample=True)
    inp = self._inps(pred['stoch'], prev_action)
    deter = self._core(prev_state['deter'], inp)
    return cast({**pred, 'deter': deter})

  def get_dist(self, stats):
    logit = stats['logit'].astype(f32)
    return tfd.Independent(jaxutils.OneHotDist(logit), 1)

  def loss(self, post, prev_state, prev_action, free=1.0):
    prev_deter = jnp.concatenate([
        prev_state['deter'][:, None], post['deter'][:, :-1]], 1)
    prior = self._pred(prev_deter, prev_action, sample=False)
    dyn = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
    rep = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
    if free:
      dyn = jnp.maximum(dyn, free)
      rep = jnp.maximum(rep, free)
    return {'dyn': dyn, 'rep': rep}, prior

  def _inps(self, stoch, act):
    batch_shape = stoch.shape[:-2]
    if self._action_clip > 0.0:
      act *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(act)))
    x = jnp.concatenate([
        cast(act).reshape((*batch_shape, -1)),
        cast(stoch).reshape((*batch_shape, -1))], -1)
    x = self.get('inps', Linear, **self._kw)(x)
    return x

  def _repr(self, embed, sample=True):
    x = self.get('repr', Linear, **self._kw)(embed)
    stats = self._stats('repr_stats', x)
    stoch = self.get_dist(stats).sample(seed=nj.rng()) if sample else None
    return cast({**stats, 'stoch': stoch})

  def _pred(self, deter, act, sample=True):
    x = jnp.concatenate([deter, act], -1)
    for i in range(self._prior_layers):
      x = self.get(f'pred{i}', Linear, **self._kw)(x)
    stats = self._stats('pred_stats', x)
    stoch = self.get_dist(stats).sample(seed=nj.rng()) if sample else None
    return cast({**stats, 'stoch': stoch})

  def _core(self, prev_deter, inputs):
    x = jnp.concatenate([prev_deter, inputs], -1)
    if self._bottleneck > 0:
      kw = {**self._kw, 'units': self._bottleneck}
      x = self.get('bottleneck', Linear, **kw)(x)
    kw = {**self._kw, 'act': 'none', 'units': 3 * self._deter}
    x = self.get('gru', Linear, **kw)(x)
    reset, cand, update = jnp.split(x, 3, -1)
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * prev_deter
    return deter

  def _stats(self, name, x):
    x = self.get(name, Linear, self._stoch * self._classes)(x)
    logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
    if self._unimix:
      probs = jax.nn.softmax(logit, -1)
      uniform = jnp.ones_like(probs) / probs.shape[-1]
      probs = (1 - self._unimix) * probs + self._unimix * uniform
      logit = jnp.log(probs)
    return {'logit': logit}


class MultiEncoder(nj.Module):

  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', mlp_layers=4,
      mlp_units=512, cnn='resize', cnn_depth=48,
      cnn_blocks=2, resize='stride',
      symlog_inputs=False, minres=4, **kw):
    excluded = ('is_first', 'is_last')
    shapes = {k: v for k, v in shapes.items() if (
        k not in excluded and not k.startswith('log_'))}
    self.cnn_shapes = {k: v for k, v in shapes.items() if (
        len(v) == 3 and re.match(cnn_keys, k))}
    self.mlp_shapes = {k: v for k, v in shapes.items() if (
        len(v) in (1, 2) and re.match(mlp_keys, k))}
    assert not ("token" in self.mlp_shapes and \
                "token_embed" in self.mlp_shapes), \
      "Probably shouldn't have both token and token_embed, use token$?"
    self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
    print('Encoder CNN shapes:', self.cnn_shapes)
    print('Encoder MLP shapes:', self.mlp_shapes)
    cnn_kw = {**kw, 'minres': minres, 'name': 'cnn'}
    mlp_kw = {**kw, 'symlog_inputs': symlog_inputs, 'name': 'mlp'}
    if cnn == 'resnet':
      self._cnn = ImageEncoderResnet(cnn_depth, cnn_blocks, resize, **cnn_kw)
    else:
      raise NotImplementedError(cnn)
    if self.mlp_shapes:
      self._mlp = MLP(None, mlp_layers, mlp_units, dist='none', **mlp_kw)
    self.preprocessors = {}

  def __call__(self, data, zero_mlp=False, zero_cnn=False):
    some_key, some_shape = list(self.shapes.items())[0]
    batch_dims = data[some_key].shape[:-len(some_shape)]
    data = {
        k: v.reshape((-1,) + v.shape[len(batch_dims):])
        for k, v in data.items()}
    outputs = []
    if self.cnn_shapes:
      inputs = jnp.concatenate([data[k] for k in self.cnn_shapes], -1)
      output = self._cnn(inputs)
      output = output.reshape((output.shape[0], -1))
      if zero_cnn:
        output = jnp.zeros_like(output)
      outputs.append(output)
    if self.mlp_shapes:
      inputs = [
          data[k][..., None] if len(self.shapes[k]) == 0 else data[k]
          for k in self.mlp_shapes]
      inputs = jnp.concatenate([x.astype(f32) for x in inputs], -1)
      inputs = jaxutils.cast_to_compute(inputs)
      output = self._mlp(inputs)
      if zero_mlp:
        output = jnp.zeros_like(output)
      outputs.append(output)
    outputs = jnp.concatenate(outputs, -1)
    outputs = outputs.reshape(batch_dims + outputs.shape[1:])
    return outputs


class MultiDecoder(nj.Module):

  def __init__(
      self, shapes, inputs=['tensor'], cnn_keys=r'.*', mlp_keys=r'.*',
      mlp_layers=4, mlp_units=512, cnn='resize', cnn_depth=48, cnn_blocks=2,
      image_dist='mse', vector_dist='mse', resize='stride', bins=255,
      outscale=1.0, minres=4, cnn_sigmoid=False, **kw):
    excluded = ('is_first', 'is_last', 'is_terminal', 'reward')
    shapes = {k: v for k, v in shapes.items() if k not in excluded}
    self.cnn_shapes = {
        k: v for k, v in shapes.items()
        if re.match(cnn_keys, k) and len(v) == 3}
    self.mlp_shapes = {
        k: v for k, v in shapes.items()
        if re.match(mlp_keys, k) and len(v) == 1}
    self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
    print('Decoder CNN shapes:', self.cnn_shapes)
    print('Decoder MLP shapes:', self.mlp_shapes)
    cnn_kw = {**kw, 'minres': minres, 'sigmoid': cnn_sigmoid}
    mlp_kw = {**kw, 'dist': vector_dist, 'outscale': outscale, 'bins': bins}
    if self.cnn_shapes:
      shapes = list(self.cnn_shapes.values())
      assert all(x[:-1] == shapes[0][:-1] for x in shapes)
      shape = shapes[0][:-1] + (sum(x[-1] for x in shapes),)
      if cnn == 'resnet':
        self._cnn = ImageDecoderResnet(
            shape, cnn_depth, cnn_blocks, resize, **cnn_kw, name='cnn')
      elif cnn == 'style':
        self._cnn = ImageDecoderStyle(
            shape, cnn_depth, cnn_blocks, resize, **cnn_kw, name='cnn')
      else:
        raise NotImplementedError(cnn)
    if self.mlp_shapes:
      self._mlp = MLP(
          self.mlp_shapes, mlp_layers, mlp_units, **mlp_kw, name='mlp')
    self._inputs = Input(inputs, dims='deter')
    self._image_dist = image_dist

  def __call__(self, inputs, drop_loss_indices=None):
    features = self._inputs(inputs)
    dists = {}
    if self.cnn_shapes:
      feat = features
      if drop_loss_indices is not None:
        feat = feat[:, drop_loss_indices]
      flat = feat.reshape([-1, feat.shape[-1]])
      output = self._cnn(flat)
      output = output.reshape(feat.shape[:-1] + output.shape[1:])
      split_indices = np.cumsum([v[-1] for v in self.cnn_shapes.values()][:-1])
      means = jnp.split(output, split_indices, -1)
      dists.update({
          key: self._make_image_dist(key, mean)
          for (key, shape), mean in zip(self.cnn_shapes.items(), means)})
    if self.mlp_shapes:
      dists.update(self._mlp(features))
    return dists

  def _make_image_dist(self, name, mean):
    mean = mean.astype(f32)
    if self._image_dist == 'normal':
      return tfd.Independent(tfd.Normal(mean, 1), 3)
    if self._image_dist == 'mse':
      return jaxutils.MSEDist(mean, 3, 'sum')
    if self._image_dist == 'mse_max':
      return jaxutils.MSEMaxDist(mean, 3, 'sum')
    if self._image_dist == 'abs':
      return jaxutils.AbsDist(mean, 3, 'sum')
    if self._image_dist == 'binary':
      return tfd.Independent(tfd.Bernoulli(mean), 3)
    raise NotImplementedError(self._image_dist)

class ImageEncoderResnet(nj.Module):

  def __init__(self, depth, blocks, resize, minres, **kw):
    self._depth = depth
    self._blocks = blocks
    self._resize = resize
    self._minres = minres
    self._kw = kw

  def __call__(self, x):
    stages = int(np.log2(x.shape[-2]) - np.log2(self._minres))
    depth = self._depth
    x = jaxutils.cast_to_compute(x) - 0.5
    # print(x.shape)
    for i in range(stages):
      kw = {**self._kw, 'preact': False}
      if self._resize == 'stride':
        x = self.get(f's{i}res', Conv2D, depth, 4, 2, **kw)(x)
      elif self._resize == 'stride3':
        s = 2 if i else 3
        k = 5 if i else 4
        x = self.get(f's{i}res', Conv2D, depth, k, s, **kw)(x)
      elif self._resize == 'mean':
        N, H, W, D = x.shape
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
        x = x.reshape((N, H // 2, W // 2, 4, D)).mean(-2)
      elif self._resize == 'max':
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
        x = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), 'same')
      else:
        raise NotImplementedError(self._resize)
      for j in range(self._blocks):
        skip = x
        kw = {**self._kw, 'preact': True}
        x = self.get(f's{i}b{j}conv1', Conv2D, depth, 3, **kw)(x)
        x = self.get(f's{i}b{j}conv2', Conv2D, depth, 3, **kw)(x)
        x += skip
        # print(x.shape)
      depth *= 2
    if self._blocks:
      x = get_act(self._kw['act'])(x)
    x = x.reshape((x.shape[0], -1))
    # print(x.shape)
    return x


class ImageDecoderResnet(nj.Module):

  def __init__(self, shape, depth, blocks, resize, minres, sigmoid, **kw):
    self._shape = shape
    self._depth = depth
    self._blocks = blocks
    self._resize = resize
    self._minres = minres
    self._sigmoid = sigmoid
    self._kw = kw

  def __call__(self, x):
    stages = int(np.log2(self._shape[-2]) - np.log2(self._minres))
    depth = self._depth * 2 ** (stages - 1)
    x = jaxutils.cast_to_compute(x)
    x = self.get('in', Linear, (self._minres, self._minres, depth))(x)

    for i in range(stages):
      for j in range(self._blocks):
        skip = x
        kw = {**self._kw, 'preact': True}
        x = self.get(f's{i}b{j}conv1', Conv2D, depth, 3, **kw)(x)
        x = self.get(f's{i}b{j}conv2', Conv2D, depth, 3, **kw)(x)
        x += skip
        # print(x.shape)
      depth //= 2
      kw = {**self._kw, 'preact': False}
      if i == stages - 1:
        kw = {}
        depth = self._shape[-1]
      if self._resize == 'stride':
        x = self.get(f's{i}res', Conv2D, depth, 4, 2, transp=True, **kw)(x)
      elif self._resize == 'stride3':
        s = 3 if i == stages - 1 else 2
        k = 5 if i == stages - 1 else 4
        x = self.get(f's{i}res', Conv2D, depth, k, s, transp=True, **kw)(x)
      elif self._resize == 'resize':
        x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
      else:
        raise NotImplementedError(self._resize)
    if max(x.shape[1:-1]) > max(self._shape[:-1]):
      padh = (x.shape[1] - self._shape[0]) / 2
      padw = (x.shape[2] - self._shape[1]) / 2
      x = x[:, int(np.ceil(padh)): -int(padh), :]
      x = x[:, :, int(np.ceil(padw)): -int(padw)]
    # print(x.shape)
    assert x.shape[-3:] == self._shape, (x.shape, self._shape)
    if self._sigmoid:
      x = jax.nn.sigmoid(x)
    else:
      x = x + 0.5
    return x


class ImageDecoderStyle(nj.Module):

  def __init__(
      self, shape, depth, blocks, resize, minres, sigmoid, **kw):
    self._shape = shape
    self._depth = depth
    self._blocks = blocks
    self._resize = resize
    self._minres = minres
    self._sigmoid = sigmoid
    self._kw = kw

  def __call__(self, x):
    stages = int(np.log2(self._shape[-2]) - np.log2(self._minres))

    style = x
    for i in range(4):
      style = self.get(f'style{i}', Linear, 1024, **self._kw)(style)

    depth = self._depth * 2 ** (stages - 1)
    x = jaxutils.cast_to_compute(x)
    x = self.get('in', Linear, (self._minres, self._minres, depth))(x)
    for i in range(stages):
      for j in range(self._blocks):
        skip = x
        kw = {**self._kw, 'preact': True}
        s1 = self.get(f's{i}b{j}s1', Linear, 2 * depth)(style)
        s2 = self.get(f's{i}b{j}s2', Linear, 2 * depth)(style)
        s1 = jnp.split(s1[..., None, None, :], 2, -1)
        s2 = jnp.split(s2[..., None, None, :], 2, -1)
        x = self.get(f's{i}b{j}c1', Conv2D, depth, 3, **kw)(x, s1)
        x = self.get(f's{i}b{j}c2', Conv2D, depth, 3, **kw)(x, s2)
        x += skip
        # print(x.shape)
      depth //= 2
      kw = {**self._kw, 'preact': False}
      if i == stages - 1:
        kw = {}
        depth = self._shape[-1]
      if self._resize == 'stride':
        s = None
        if self._blocks == 0:
          s = self.get(f's{i}s', Linear, 2 * depth)(style)
          s = jnp.split(s[..., None, None, :], 2, -1)
        x = self.get(f's{i}res', Conv2D, depth, 4, 2, transp=True, **kw)(x, s)
      elif self._resize == 'stride3':
        s = 3 if i == stages - 1 else 2
        k = 5 if i == stages - 1 else 4
        x = self.get(f's{i}res', Conv2D, depth, k, s, transp=True, **kw)(x)
      elif self._resize == 'resize':
        x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
      else:
        raise NotImplementedError(self._resize)
    if max(x.shape[1:-1]) > max(self._shape[:-1]):
      padh = (x.shape[1] - self._shape[0]) / 2
      padw = (x.shape[2] - self._shape[1]) / 2
      x = x[:, int(np.ceil(padh)): -int(padh), :]
      x = x[:, :, int(np.ceil(padw)): -int(padw)]
    # print(x.shape)
    assert x.shape[-3:] == self._shape, (x.shape, self._shape)
    if self._sigmoid:
      x = jax.nn.sigmoid(x)
    else:
      x = x + 0.5
    return x


class MLP(nj.Module):

  def __init__(
      self, shape, layers, units, inputs=['tensor'], dims=None,
      symlog_inputs=False, **kw):
    assert shape is None or isinstance(shape, (int, tuple, dict)), shape
    if isinstance(shape, int):
      shape = (shape,)
    self._shape = shape
    self._layers = layers
    self._units = units
    self._inputs = Input(inputs, dims=dims)
    self._symlog_inputs = symlog_inputs
    distkeys = (
        'dist', 'outscale', 'minstd', 'maxstd', 'outnorm', 'unimix', 'bins')
    self._dense = {k: v for k, v in kw.items() if k not in distkeys}
    self._dist = {k: v for k, v in kw.items() if k in distkeys}

  def __call__(self, inputs):
    feat = self._inputs(inputs)
    if self._symlog_inputs:
      feat = jaxutils.symlog(feat)
    x = jaxutils.cast_to_compute(feat)
    x = x.reshape([-1, x.shape[-1]])
    for i in range(self._layers):
      x = self.get(f'h{i}', Linear, self._units, **self._dense)(x)
    x = x.reshape(feat.shape[:-1] + (x.shape[-1],))
    if self._shape is None:
      return x
    elif isinstance(self._shape, tuple):
      return self._out('out', self._shape, x)
    elif isinstance(self._shape, dict):
      return {k: self._out(k, v, x) for k, v in self._shape.items()}
    else:
      raise ValueError(self._shape)

  def _out(self, name, shape, x):
    return self.get(f'dist_{name}', Dist, shape, **self._dist)(x)


class Dist(nj.Module):

  def __init__(
      self, shape, dist='mse', outscale=0.1, outnorm=False, minstd=1.0,
      maxstd=1.0, unimix=0.0, bins=255):
    assert all(isinstance(dim, int) for dim in shape), shape
    self._shape = shape
    self._dist = dist
    self._minstd = minstd
    self._maxstd = maxstd
    self._unimix = unimix
    self._outscale = outscale
    self._outnorm = outnorm
    self._bins = bins

  def __call__(self, inputs):
    dist = self.inner(inputs)
    assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
        dist.batch_shape, dist.event_shape, inputs.shape)
    return dist

  def inner(self, inputs):
    kw = {}
    kw['outscale'] = self._outscale
    kw['outnorm'] = self._outnorm
    shape = self._shape
    if self._dist.endswith('_twohot'):
      shape = (*self._shape, self._bins)
    out = self.get('out', Linear, int(np.prod(shape)), **kw)(inputs)
    out = out.reshape(inputs.shape[:-1] + shape).astype(f32)
    if self._dist in ('normal', 'trunc_normal'):
      std = self.get('std', Linear, int(np.prod(self._shape)), **kw)(inputs)
      std = std.reshape(inputs.shape[:-1] + self._shape).astype(f32)
    if self._dist == 'symlog_mse':
      return jaxutils.SymlogDist(out, len(self._shape), 'mse', 'sum')
    if self._dist == 'symlog_and_twohot':
      bins = np.linspace(-20, 20, out.shape[-1])
      return jaxutils.TwoHotDist(
          out, bins, len(self._shape), jaxutils.symlog, jaxutils.symexp)
    if self._dist == 'symexp_twohot':
      bins = jaxutils.symexp(np.linspace(-20, 20, out.shape[-1]))
      return jaxutils.TwoHotDist(out, bins, len(self._shape))
    if self._dist == 'parab_twohot':
      eps = 0.001
      f = lambda x: np.sign(x) * (np.square(np.sqrt(
          1 + 4 * eps * (eps + 1 + np.abs(x))) / 2 / eps - 1 / 2 / eps) - 1)
      bins = f(np.linspace(-300, 300, out.shape[-1]))
      return jaxutils.TwoHotDist(out, bins, len(self._shape))
    if self._dist == 'mse':
      return jaxutils.MSEDist(out, len(self._shape), 'sum')
    if self._dist == 'normal':
      lo, hi = self._minstd, self._maxstd
      std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
      dist = tfd.Normal(jnp.tanh(out), std)
      dist = tfd.Independent(dist, len(self._shape))
      dist.minent = np.prod(self._shape) * tfd.Normal(0.0, lo).entropy()
      dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
      return dist
    if self._dist == 'binary':
      dist = tfd.Bernoulli(out)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'onehot':
      if self._unimix:
        probs = jax.nn.softmax(out, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        out = jnp.log(probs)
      dist = jaxutils.OneHotDist(out)
      if len(self._shape) > 1:
        dist = tfd.Independent(dist, len(self._shape) - 1)
      dist.minent = 0.0
      dist.maxent = np.prod(self._shape[:-1]) * jnp.log(self._shape[-1])
      return dist
    raise NotImplementedError(self._dist)


class VectorQuantizer(nj.Module):

  def __init__(self, codes=512, embed=32):
    self.codes = codes
    self.book = nj.Variable(lambda: jax.random.normal(
        nj.rng(), (self.codes, embed), jnp.float32))

  def __call__(self, inputs):
    book = self.book.read()
    book /= jnp.linalg.norm(book, 2, -1, True)
    flat = inputs.reshape((-1, inputs.shape[-1]))
    flat /= jnp.linalg.norm(flat, 2, -1, True)
    flat2 = (flat ** 2).sum(-1, keepdims=True)
    book2 = (book ** 2).sum(-1, keepdims=True).T
    dist = flat2 - 2 * (flat @ book.T) + book2
    indices = jnp.argmin(dist, -1).reshape(inputs.shape[:-1])
    outputs = book[indices]
    outputs = inputs + sg(outputs - inputs)
    return outputs, indices

  def embed(self, indices):
    book = self.book.read()
    book /= jnp.linalg.norm(book, 2, -1, True)
    return book[indices]

  def loss(self, inputs, indices, beta=0.25):
    inputs = inputs.astype(jnp.float32)
    embed = self.embed(indices).astype(jnp.float32)
    loss_enc  = ((sg(embed) - inputs) ** 2).mean(-1)
    loss_book = ((embed - sg(inputs)) ** 2).mean(-1)
    return loss_enc + beta * loss_book


class Block(nj.Module):

  def __init__(
      self, size, groups=8, heads=8, act='gelu', norm='layer',
      winit='normal', fan='avg'):
    assert norm == 'layer', norm
    assert size % groups == 0, (size, groups)
    assert (size // groups) % heads == 0, (size, groups, heads)
    self.size = size
    self.act = get_act(act)
    self.groups = groups
    self.heads = heads
    self.kw = dict(winit=winit, fan=fan)

  def __call__(self, x):
    if x.shape[-1] % self.groups != 0:
      want = int(np.ceil(x.shape[-1] / self.groups) * self.groups)
      missing = want - x.shape[-1]
      x = jnp.concatenate([x, x[..., :missing]], -1)
      assert x.shape[-1] % self.groups == 0, (should, x.shape, self.groups)
    embed = self.size // self.groups
    x = x.reshape((*x.shape[:-1], self.groups, x.shape[-1] // self.groups))
    if x.shape[-1] != embed:
      x = self.get('proj', Linear, embed, **self.kw)(x)
    skip = x
    x = self.get('norm1', Norm, 'layer')(x)
    dim = embed // self.heads
    x = self.get('attn1', Attention, self.heads, dim, **self.kw)(x, x, x)
    x += skip
    skip = x
    x = self.get('norm2', Norm, 'layer')(x)
    x = self.get('linear1', Linear, embed, **self.kw)(x)
    x = self.act(x)
    x = self.get('linear2', Linear, embed, **self.kw)(x)
    x += skip
    x = x.reshape((*x.shape[:-2], self.size))
    return x


class Attention(nj.Module):

  def __init__(self, heads, size, winit='normal', fan='avg'):
    self.heads = heads
    self.size = size
    self.kw = dict(winit=winit, fan=fan)

  def __call__(self, query, key, value, mask=None):
    shape = (self.heads, self.size)
    query = self.get('query', Linear, shape, **self.kw)(query)
    key = self.get('key', Linear, shape, **self.kw)(key)
    value = self.get('value', Linear, shape, **self.kw)(value)
    logits = jnp.einsum('...thd,...Thd->...htT', query, key)
    logits /= np.sqrt(self.size).astype(key.dtype)
    if mask is not None:
      assert mask.ndim == logits.ndim
      logits = jnp.where(mask, logits, -np.inf)
    weights = jax.nn.softmax(logits)
    x = jnp.einsum('...htT,...Thd->...thd', weights, value)
    x = x.reshape((*x.shape[:-2], -1))
    x = self.get('out', Linear, self.heads * self.size)(x)
    return x


class Conv2D(nj.Module):

  def __init__(
      self, depth, kernel, stride=1, transp=False, act='none', norm='none',
      pad='same', bias=True, preact=False, winit='uniform', fan='avg'):
    self._depth = depth
    self._kernel = kernel
    self._stride = stride
    self._transp = transp
    self._act = get_act(act)
    self._norm = Norm(norm, name='norm')
    self._pad = pad.upper()
    self._bias = bias and (preact or norm == 'none')
    self._preact = preact
    self._winit = winit
    self._fan = fan

  def __call__(self, hidden, style=None):
    if self._preact:
      hidden = self._norm(hidden, style)
      hidden = self._act(hidden)
      hidden = self._layer(hidden)
    else:
      hidden = self._layer(hidden)
      hidden = self._norm(hidden, style)
      hidden = self._act(hidden)
    return hidden

  def _layer(self, x):
    if self._transp:
      shape = (self._kernel, self._kernel, self._depth, x.shape[-1])
      kernel = self.get('kernel', Initializer(
          self._winit, fan=self._fan), shape)
      kernel = jaxutils.cast_to_compute(kernel)
      x = jax.lax.conv_transpose(
          x, kernel, (self._stride, self._stride), self._pad,
          dimension_numbers=('NHWC', 'HWOI', 'NHWC'))
    else:
      shape = (self._kernel, self._kernel, x.shape[-1], self._depth)
      kernel = self.get('kernel', Initializer(
          self._winit, fan=self._fan), shape)
      kernel = jaxutils.cast_to_compute(kernel)
      x = jax.lax.conv_general_dilated(
          x, kernel, (self._stride, self._stride), self._pad,
          dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    if self._bias:
      bias = self.get('bias', jnp.zeros, self._depth, np.float32)
      bias = jaxutils.cast_to_compute(bias)
      x += bias
    return x


class Linear(nj.Module):

  def __init__(
      self, units, act='none', norm='none', bias=True, outscale=1.0,
      outnorm=False, winit='normal', fan='avg'):
    self._units = tuple(units) if hasattr(units, '__len__') else (units,)
    self._act = get_act(act)
    self._norm = norm
    self._bias = bias and norm == 'none'
    self._outscale = outscale
    self._outnorm = outnorm
    self._winit = winit
    self._fan = fan

  def __call__(self, x):
    shape = (x.shape[-1], np.prod(self._units))
    kernel = self.get('kernel', Initializer(
        self._winit, self._outscale, fan=self._fan), shape)
    kernel = jaxutils.cast_to_compute(kernel)
    x = x @ kernel
    if self._bias:
      bias = self.get('bias', jnp.zeros, np.prod(self._units), np.float32)
      bias = jaxutils.cast_to_compute(bias)
      x += bias
    if len(self._units) > 1:
      x = x.reshape(x.shape[:-1] + self._units)
    x = self.get('norm', Norm, self._norm)(x)
    x = self._act(x)
    return x


class Norm(nj.Module):

  def __init__(self, impl):
    self._impl = impl

  def __call__(self, x, style=None):
    dtype = x.dtype
    if self._impl == 'none':
      return x
    elif self._impl == 'layer':
      x = x.astype(f32)
      x = jax.nn.standardize(x, axis=-1, epsilon=1e-3)
      if style is None:
        x *= self.get('scale', jnp.ones, x.shape[-1], f32)
        x += self.get('bias', jnp.zeros, x.shape[-1], f32)
      else:
        x *= style[0]
        x += style[1]
      return x.astype(dtype)
    else:
      raise NotImplementedError(self._impl)


class Input:

  def __init__(self, keys=['tensor'], dims=None):
    assert isinstance(keys, (list, tuple)), keys
    self._keys = tuple(keys)
    self._dims = dims or self._keys[0]

  def __call__(self, inputs):
    if not isinstance(inputs, dict):
      inputs = {'tensor': inputs}
    inputs = inputs.copy()
    for key in self._keys:
      if key.startswith('softmax_'):
        inputs[key] = jax.nn.softmax(inputs[key[len('softmax_'):]])
    if not all(k in inputs for k in self._keys):
      needs = f'{{{", ".join(self._keys)}}}'
      found = f'{{{", ".join(inputs.keys())}}}'
      raise KeyError(f'Cannot find keys {needs} among inputs {found}.')
    values = [inputs[k] for k in self._keys]
    dims = len(inputs[self._dims].shape)
    for i, value in enumerate(values):
      if len(value.shape) > dims:
        values[i] = value.reshape(
            value.shape[:dims - 1] + (np.prod(value.shape[dims - 1:]),))
    values = [x.astype(inputs[self._dims].dtype) for x in values]
    return jnp.concatenate(values, -1)


class Initializer:

  def __init__(self, dist='uniform', scale=1.0, fan='avg'):
    self.scale = scale
    self.dist = dist
    self.fan = fan

  def __call__(self, shape):
    if self.scale == 0.0:
      value = jnp.zeros(shape, f32)
    elif self.dist == 'uniform':
      fanin, fanout = self._fans(shape)
      denoms = {'avg': (fanin + fanout) / 2, 'in': fanin, 'out': fanout}
      scale = self.scale / denoms[self.fan]
      limit = np.sqrt(3 * scale)
      value = jax.random.uniform(
          nj.rng(), shape, f32, -limit, limit)
    elif self.dist == 'normal':
      fanin, fanout = self._fans(shape)
      denoms = {'avg': np.mean((fanin, fanout)), 'in': fanin, 'out': fanout}
      scale = self.scale / denoms[self.fan]
      std = np.sqrt(scale) / 0.87962566103423978
      value = std * jax.random.truncated_normal(
          nj.rng(), -2, 2, shape, f32)
    elif self.dist == 'ortho':
      nrows, ncols = shape[-1], np.prod(shape) // shape[-1]
      matshape = (nrows, ncols) if nrows > ncols else (ncols, nrows)
      mat = jax.random.normal(nj.rng(), matshape, f32)
      qmat, rmat = jnp.linalg.qr(mat)
      qmat *= jnp.sign(jnp.diag(rmat))
      qmat = qmat.T if nrows < ncols else qmat
      qmat = qmat.reshape(nrows, *shape[:-1])
      value = self.scale * jnp.moveaxis(qmat, 0, -1)
    else:
      raise NotImplementedError(self.dist)
    return value

  def _fans(self, shape):
    if len(shape) == 0:
      return 1, 1
    elif len(shape) == 1:
      return shape[0], shape[0]
    elif len(shape) == 2:
      return shape
    else:
      space = int(np.prod(shape[:-2]))
      return shape[-2] * space, shape[-1] * space


def get_act(name):
  if callable(name):
    return name
  elif name == 'none':
    return lambda x: x
  elif name == 'mish':
    return lambda x: x * jnp.tanh(jax.nn.softplus(x))
  elif name == 'gelu2':
    return lambda x: jax.nn.sigmoid(1.702 * x) * x
  elif hasattr(jax.nn, name):
    return getattr(jax.nn, name)
  else:
    raise NotImplementedError(name)
