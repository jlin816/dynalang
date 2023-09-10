import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
f32 = jnp.float32
tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

from . import jaxutils
from . import nets
from . import ninjax as nj
cast = jaxutils.cast_to_compute


class S5RSSM(nj.Module):

  def __init__(
      self, state=256, hidden=256, layers=10, stoch=32, classes=32,
      unimix=0.01, action_clip=1.0, **kw):
    self._state = state
    self._hidden = hidden
    self._layers = layers
    self._stoch = stoch
    self._classes = classes
    self._unimix = unimix
    self._action_clip = action_clip
    self._kw = kw
    self._ssm = S5Model(state, hidden, layers, name='s5')

  def initial(self, batch_size):
    return dict(
        deter=self._ssm.initial(batch_size),
        out=cast(jnp.zeros((batch_size, self._hidden), f32)),
        stoch=cast(jnp.zeros((batch_size, self._stoch, self._classes), f32)),
        logit=cast(jnp.zeros((batch_size, self._stoch, self._classes), f32)),
    )

  def observe(self, state, actions, embeds, resets):
    B, T, _ = embeds.shape
    actions = cast(jaxutils.concat_dict(actions))
    actions = jaxutils.switch(resets, jnp.zeros_like(actions), actions)
    stoch, logit = self._rep(embeds)
    prev_actions = actions
    prev_stoch = jnp.concatenate([state['stoch'][:, None], stoch[:, :-1]], 1)
    inps = self._inp(prev_actions, prev_stoch)
    out, deter = self._ssm(inps, resets, state['deter'])
    states = dict(
        deter=deter, out=cast(out),
        stoch=cast(stoch), logit=cast(logit))
    return states

  def imagine(self, state, actions):
    return jaxutils.scan(self.img_step, actions, state, axis=1)

  def obs_step(self, state, action, embed, reset):
    inputs = tree_map(lambda x: x[:, None], (action, embed, reset))
    states = self.observe(state, *inputs)
    state = tree_map(lambda x: x[:, -1], states)
    return state

  def img_step(self, state, action):
    action = cast(jaxutils.concat_dict(action))
    resets = jnp.zeros(len(action), bool)[:, None]
    inps = self._inp(action[:, None], state['stoch'][:, None])
    out, deter = self._ssm(inps, resets, state['deter'])
    logit = self._logit('prior', out)
    stoch = self._dist(logit).sample(seed=nj.rng())
    states = dict(
        deter=deter, out=cast(out),
        stoch=cast(stoch), logit=cast(logit))
    state = {k: v[:, -1] for k, v in states.items()}
    return state

  def loss(self, obs_out, free=1.0):
    metrics = {}
    prior = self._logit('prior', obs_out['out'])
    post = obs_out['logit']
    dyn = self._dist(sg(post)).kl_divergence(self._dist(prior))
    rep = self._dist(post).kl_divergence(self._dist(sg(prior)))
    if free:
      dyn = jnp.maximum(dyn, free)
      rep = jnp.maximum(rep, free)
    losses = {'dyn': dyn, 'rep': rep}
    metrics['prior_ent'] = self._dist(prior).entropy()
    metrics['post_ent'] = self._dist(post).entropy()
    return losses, metrics

  def _rep(self, embed):
    x = self.get('rep', nets.Linear, self._hidden, **self._kw)(embed)
    logit = self._logit('rep_logit', x)
    stoch = self._dist(logit).sample(seed=nj.rng())
    return stoch, logit

  def _inp(self, act, stoch):
    bs = stoch.shape[:-2]
    if self._action_clip > 0.0:
      act *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(act)))
    x = jnp.concatenate([
        cast(act).reshape((*bs, -1)),
        cast(stoch).reshape((*bs, -1)),
    ], -1)
    x = self.get('inp', nets.Linear, self._state, **self._kw)(x)
    return x

  def _logit(self, name, x):
    x = self.get(name, nets.Linear, self._stoch * self._classes)(x)
    logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
    if self._unimix:
      probs = jax.nn.softmax(logit, -1)
      uniform = jnp.ones_like(probs) / probs.shape[-1]
      probs = (1 - self._unimix) * probs + self._unimix * uniform
      logit = jnp.log(probs)
    return logit

  def _dist(self, logit):
    return tfd.Independent(jaxutils.OneHotDist(logit.astype(f32)), 1)


class S5DoubleRSSM(nj.Module):

  def __init__(
      self, state=256, hidden=256, layers=10, stoch=32, classes=32,
      unimix=0.01, action_clip=1.0, **kw):
    self._state = state
    self._hidden = hidden
    self._layers = layers
    self._stoch = stoch
    self._classes = classes
    self._unimix = unimix
    self._action_clip = action_clip
    self._kw = kw
    self._ssm = S5Model(state, hidden, layers, name='s5')

  def initial(self, batch_size):
    bs = batch_size
    return dict(
        state=self._ssm.initial(batch_size),
        feat1=cast(jnp.zeros((bs, self._hidden), f32)),
        feat2=cast(jnp.zeros((bs, self._hidden), f32)),
        stoch=cast(jnp.zeros((bs, self._stoch, self._classes), f32)),
        logit=cast(jnp.zeros((bs, self._stoch, self._classes), f32)),
    )

  def observe(self, state, actions, embeds, resets):
    B, T, _ = embeds.shape
    actions = cast(jaxutils.concat_dict(actions))
    actions = jaxutils.switch(resets, jnp.zeros_like(actions), actions)
    stoch, logit = self._rep(embeds)
    inp_a = self._inp_a(actions)[:, :, None, :]
    inp_z = self._inp_z(stoch)[:, :, None, :]
    inps = jnp.stack([inp_a, inp_z], 2).reshape((B, 2 * T, self._state))
    resets = jnp.stack([resets, jnp.zeros_like(resets)]).reshape((B, 2 * T))
    y, states = self._ssm(inps, resets, state['state'])
    states, feat1, feat2 = states[:, 1::2], y[:, :-1:2], y[:, 1::2]
    states = dict(
        state=states, feat1=cast(feat1), feat2=cast(feat2),
        stoch=cast(stoch), logit=cast(logit))
    return states

  def imagine(self, state, actions):
    return jaxutils.scan(self.img_step, actions, state, axis=1)

  def obs_step(self, state, action, embed, reset):
    inputs = tree_map(lambda x: x[:, None], (action, embed, reset))
    states = self.observe(state, *inputs)
    state = tree_map(lambda x: x[:, -1], states)
    return state

  def img_step(self, state, action):
    action = cast(jaxutils.concat_dict(action))
    reset = jnp.zeros(len(action), bool)[:, None]
    inp_a = self._inp_a(action[:, None])
    feat1, states1 = self._ssm(inp_a, reset, state['state'])
    logit = self._logit('prior', feat1)
    stoch = self._dist(logit).sample(seed=nj.rng())
    inp_z = self._inp_z(stoch)
    feat2, states2 = self._ssm(inp_z, reset, states1[:, -1])
    states = dict(
        state=states2, feat1=cast(feat1), feat2=cast(feat2),
        stoch=cast(stoch), logit=cast(logit))
    state = {k: v[:, -1] for k, v in states.items()}
    return state

  def loss(self, obs_out, free=1.0):
    metrics = {}
    prior = self._logit('prior', obs_out['feat1'])
    post = obs_out['logit']
    dyn = self._dist(sg(post)).kl_divergence(self._dist(prior))
    rep = self._dist(post).kl_divergence(self._dist(sg(prior)))
    if free:
      dyn = jnp.maximum(dyn, free)
      rep = jnp.maximum(rep, free)
    losses = {'dyn': dyn, 'rep': rep}
    metrics['prior_ent'] = self._dist(prior).entropy()
    metrics['post_ent'] = self._dist(post).entropy()
    return losses, metrics

  def _rep(self, embed):
    x = self.get('rep', nets.Linear, self._hidden, **self._kw)(embed)
    logit = self._logit('rep_logit', x)
    stoch = self._dist(logit).sample(seed=nj.rng())
    return stoch, logit

  def _inp_z(self, stoch):
    x = cast(stoch).reshape((*stoch.shape[:-2], -1))
    x = self.get('inp_z', nets.Linear, self._state, **self._kw)(x)
    return x

  def _inp_a(self, act):
    if self._action_clip > 0.0:
      act *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(act)))
    x = self.get('inp_a', nets.Linear, self._state, **self._kw)(act)
    return x

  def _logit(self, name, x):
    x = self.get(name, nets.Linear, self._stoch * self._classes)(x)
    logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
    if self._unimix:
      probs = jax.nn.softmax(logit, -1)
      uniform = jnp.ones_like(probs) / probs.shape[-1]
      probs = (1 - self._unimix) * probs + self._unimix * uniform
      logit = jnp.log(probs)
    return logit

  def _dist(self, logit):
    return tfd.Independent(jaxutils.OneHotDist(logit.astype(f32)), 1)


class S5Model(nj.Module):

  def __init__(self, state=256, hidden=256, layers=10, prenorm=False):
    self._state = state
    self._hidden = hidden
    self._layers = layers
    self._prenorm = prenorm

  def initial(self, batch):
    return jnp.zeros((batch, self._layers, self._state), jnp.complex64)

  def __call__(self, x, reset, state):
    new_states = []
    for i in range(self._layers):
      h = x
      if self._prenorm:
        h = self.get(f'norm{i}', nets.Norm, 'layer', 1e-9)(h)
      h, states = self.get(
          f'ssm{i}', S5Layer, self._state)(h, state[:, i], reset)
      new_states.append(states)
      # h = dropout(h)
      h = self.get(
          f'lin{i}', nets.Linear, 2 * self._hidden,
          winit='uniform', fan='avg')(h)
      h = jax.nn.glu(h)
      # h = dropout(h)
      x = x + h  # TODO: This sometimes fails?
      if not self._prenorm:
        x = self.get(f'norm{i}', nets.Norm, 'layer', 1e-9)(x)
    if self._prenorm:
      x = self.get(f'norm{i+1}', nets.Norm, 'layer', 1e-9)(x)
    new_states = jnp.stack(new_states, axis=-2)
    return x, new_states


class S5Layer(nj.Module):

  def __init__(self, state_size):
    self._state_size = state_size
    self._diag, self._v = hippo_legs_normal(state_size)
    self._v_inv = self._v.conj().T

  def initial(self, batch):
    return jnp.zeros((batch, self._state_size), jnp.complex64)

  def __call__(self, x, state, reset):
    assert len(x.shape) == 3, ('B x T x D', x.shape)
    _, input_len, in_dims = x.shape
    params = self.params(in_dims)

    def fn(params, inseq, state, reset):
      diag, b_tilde, c_tilde, d, log_step_size = params
      diag_bar, b_bar = discretize(diag, b_tilde, jnp.exp(log_step_size))
      x, states = apply_ssm(
          diag_bar, b_bar, c_tilde, d, inseq, state, reset)
      return jax.nn.gelu(x), states

    output, states = jax.vmap(fn, (None, 0, 0, 0))(params, x, state, reset)
    return output.real, states

  def params(self, in_dims):
    diag = self._get_complex(
        'diag', jnp.broadcast_to(self._diag, [self._state_size]))
    # diag = jnp.broadcast_to(self._diag, [self._state_size])

    b_mat = nets.Initializer('normal_complex', 1.0, 'in')(
        (self._state_size, in_dims), jnp.complex64)
    v_inv_b = self._v_inv @ b_mat
    b_tilde = self._get_complex('b_tilde', v_inv_b)

    c_mat_ = nets.Initializer('normal_complex', 1.0, 'in')(
        (in_dims, self._state_size, 2), jnp.complex64)
    c_mat = c_mat_[..., 0] + 1j * c_mat_[..., 1]
    c_v = c_mat @ self._v
    c_tilde = self._get_complex('c_tilde', c_v)

    d = self.get(
        'd', nets.Initializer('normal', 1.0, 'in'), [in_dims], jnp.float32)

    log_step_sizes = self.get(
        'log_step_sizes', jax.random.uniform, nj.rng(),
        [self._state_size], jnp.float32, np.log(0.001), np.log(0.1))

    return diag, b_tilde, c_tilde, d, log_step_sizes

  def _get_complex(self, name, value):
    real = self.get(f'{name}_r', jnp.array, value.real, jnp.float32)
    imag = self.get(f'{name}_i', jnp.array, value.imag, jnp.float32)
    return real + imag * 1j


def hippo_legs_normal(size):
  assert size >= 2, size
  v = np.arange(1, size + 1)
  v2 = v * 2 + 1
  m = np.diag(v) - np.sqrt(np.outer(v2, v2))
  hippo = np.tril(m)
  p = 0.5 * np.sqrt(2 * np.arange(1, size + 1) + 1.0)
  q = 2 * p
  s = hippo + p[:, None] * q[None, :]
  diag, eig_vecs = np.linalg.eig(s)
  return np.array(diag), np.array(eig_vecs)


def discretize(diag, b_tilde, step_size):
  ident = jnp.ones(diag.shape[0])
  diag_bar = jnp.exp(diag * step_size)
  b_bar = (1 / diag * (diag_bar - ident))[..., None] * b_tilde
  return diag_bar, b_bar


def binary_operator(e_i, e_j):
  a_i, bu_i, r_i = e_i
  a_j, bu_j, r_j = e_j
  return (
      r_j * (a_j) + (1 - r_j) * (a_j * a_i),
      r_j * bu_j + (1 - r_j) * (a_j * bu_i + bu_j),
      r_j + (1 - r_j) * r_i,
  )


def apply_ssm(diag_bar, b_bar, c_tilde, d, inseq, state, reset):
  diag_elements = jnp.repeat(diag_bar[None, ...], inseq.shape[0], axis=0)
  bu_elements = jax.vmap(lambda u: b_bar @ u)(inseq)
  init_diag = jnp.ones_like(diag_bar)[None, ...]
  init_bu = state[None, ...]
  init_reset = jnp.zeros_like(reset)[0, None, ...]
  diag_elements = jnp.concatenate([init_diag, diag_elements], axis=0)
  bu_elements = jnp.concatenate([init_bu, bu_elements], axis=0)
  reset = jnp.concatenate(
      [init_reset, reset], axis=0, dtype=jnp.float32)[..., None]
  elements = (diag_elements, bu_elements, reset)  # (L, P), (L, P), (L, 1)
  _, xs, _ = jax.lax.associative_scan(binary_operator, elements)  # (L, P)
  xs = xs[1:]
  ys = jax.vmap(lambda x, u: (c_tilde @ x + d * u).real)(xs, inseq)
  # return ys, xs[-1]  # Returs only the last carry state.
  return ys, xs
