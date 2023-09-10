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


class MaskGit(nj.Module):

  def __init__(
      self, stoch, classes, embed, layers, heads, ffwdim, steps,
      **kw):
    self._stoch = stoch
    self._classes = classes
    self._embed_dim = embed
    self._num_layers = layers
    self._num_heads = heads
    self._ffw_dim = ffwdim
    self._mask_id = classes
    self._T = steps
    self._kw = kw

  def _schedule(self, ratio, method='cosine'):
    if method == 'uniform':
      mask_ratio = 1. - ratio
    elif 'pow' in method:
      exponent = float(method.replace('pow', ''))
      mask_ratio = 1. - ratio ** exponent
    elif method == 'cosine':
      mask_ratio = jax.lax.cos(np.pi / 2. * ratio)
    elif method == 'log':
      mask_ratio = -jnp.log2(ratio) / jnp.log2(self._stoch)
    elif method == 'exp':
      mask_ratio = 1 - jnp.exp2(-jnp.log2(self._stoch) * (1 - ratio))
    mask_ratio = jnp.clip(mask_ratio, 1e-6, 1.)
    return mask_ratio

  @property
  def _mask_token(self):
    token = jax.nn.one_hot(self._classes, num_classes=self._classes + 1)
    return token[None, None]

  def __call__(self, z, mask, h):
    h = self.get(
        'h_dense', nets.Linear, (self._stoch, self._embed_dim), **self._kw)(h)

    masked_z = jnp.where(mask, self._mask_token, z)
    embeddings = self.get(
        'embeddings',
        nets.Initializer(dist='normal', fan='in'),
        (self._classes + 1, self._embed_dim))
    masked_z = masked_z @ embeddings
    # TODO process h correctly
    x = self.get(
        'tfm', Transformer, self._num_layers, self._num_heads, self._ffw_dim,
        **self._kw)(masked_z, cond=h)
    x = self.get('mlm_dense', nets.Linear, self._embed_dim, **self._kw)(x)
    x = nets.get_act('gelu2')(x)
    x = self.get('mlm_norm', nets.Norm, 'layer')(x)

    logits = x @ jnp.transpose(embeddings[:-1])
    bias = self.get('mlm_bias', jnp.zeros, self._classes, jnp.float32)
    logits += bias
    return logits

  def train(self, z, h):
    # z: BLC, h: BD
    B = z.shape[0]

    ratio = jax.random.uniform(nj.rng(), shape=(B,), dtype=jnp.float32)
    ratio = self._schedule(ratio, method='cosine')
    ratio = jnp.maximum(1, jnp.floor(ratio * self._stoch))

    sample = jnp.arange(self._stoch)[None, :].repeat(B, axis=0)
    sample = jax.random.permutation(
        nj.rng(), sample, axis=-1, independent=True)
    mask = sample < ratio[:, None]

    # pad zeros to account for extra dim for mask token
    z = jnp.pad(z, ((0, 0), (0, 0), (0, 1)))
    logits = self(z, mask[..., None], h)
    return logits, mask

  def _sample_masks(self):
    idxs = jnp.arange(self._stoch, dtype=jnp.int32)
    idxs = jax.random.permutation(nj.rng(), idxs)
    chunks = jnp.array_split(idxs, self._T)

    masks = []
    for t in range(self._T):
      mask = jax.nn.one_hot(chunks[t], self._stoch).sum(axis=0).astype(bool)
      masks.append(mask[..., None])
    return masks

  def sample(self, h):
    samples = jnp.zeros(
        (h.shape[0], self._stoch, self._classes + 1), dtype=f32)
    samples = samples.at[:, :, -1].set(1.)
    mask_token = self._mask_token

    def _update(samples, masks):
      for mask in masks:
        samples = jnp.where(mask, mask_token, samples)
        logits = self(samples, mask, h)
        s = jax.random.categorical(nj.rng(), logits, axis=-1)
        s = jax.nn.one_hot(s, num_classes=self._classes + 1)
        samples = jnp.where(mask, s, samples)
      return samples

    masks = self._sample_masks()
    samples = _update(samples, masks)

    # remove extra mask token dim in one-hot
    return samples[:, :, :-1]


class Transformer(nj.Module):

  def __init__(self, num_layers, num_heads, ffw_dim, **kw):
    self._num_layers = num_layers
    self._num_heads = num_heads
    self._ffw_dim = ffw_dim
    self._kw = kw

  def __call__(self, x, cond=None, mask=None):
    x = self.get('pe', PositionEmbeddings, 'absolute')(x)
    shape = x.shape[1:-1]
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    for i in range(self._num_layers):
      x += cond
      x = self.get(
          f'layer{i}', TransformerBlock, self._num_heads, self._ffw_dim,
          **self._kw)(x, mask=mask)
    x = x.reshape(x.shape[0], *shape, x.shape[-1])
    return x


class PositionEmbeddings(nj.Module):

  def __init__(self, impl, winit='normal', fan='in', outscale=1.0):
    self._impl = impl
    self._outscale = outscale
    self._winit = winit
    self._fan = fan

  def __call__(self, x):
    if self._impl == 'absolute':
      pos_embeds = self.get('pe', nets.Initializer(
          self._winit, self._outscale, fan=self._fan), x.shape[1:])
      pos_embeds = jaxutils.cast_to_compute(pos_embeds)
    else:
      return NotImplementedError(self._impl)
    x += pos_embeds
    return x


class TransformerBlock(nj.Module):

  def __init__(self, num_heads, ffw_dim, **kw):
    self._num_heads = num_heads
    self._ffw_dim = ffw_dim
    self._kw = kw

  def __call__(self, x, mask=None):
    assert x.shape[-1] % self._num_heads == 0, (x.shape, self._num_heads)
    h = self.get('attn_norm', nets.Norm, 'layer')(x)
    h = self.get(
        'attn', MultiHeadAttention, self._num_heads,
        h.shape[-1] // self._num_heads, **self._kw)(h, h, mask=mask)
    x += h

    h = self.get('ffw_norm', nets.Norm, 'layer')(x)
    h = self.get('ffw', FFWBlock, self._ffw_dim, **self._kw)(h)
    x += h

    return x


class FFWBlock(nj.Module):

  def __init__(self, ffw_dim, **kw):
    self._ffw_dim = ffw_dim
    self._kw = kw

  def __call__(self, x):
    h = self.get('ffw1', nets.Linear, self._ffw_dim, **self._kw)(x)
    h = nets.get_act('gelu2')(h)
    h = self.get('ffw2', nets.Linear, x.shape[-1], **self._kw)(h)
    return h


class MultiHeadAttention(nj.Module):

  def __init__(self, num_heads, head_dim, **kw):
    self._num_heads = num_heads
    self._head_dim = head_dim
    self._kw = kw

  def __call__(self, inputs_q, inputs_kv, mask=None):
    outshape = (self._num_heads, self._head_dim)
    q = self.get('q', nets.Linear, outshape, **self._kw)(inputs_q)
    k = self.get('k', nets.Linear, outshape, **self._kw)(inputs_kv)
    v = self.get('v', nets.Linear, outshape, **self._kw)(inputs_kv)

    dtype = q.dtype
    scaling = np.sqrt(np.sqrt(self._head_dim))
    attn_weights = jnp.einsum(
        '...qhd,...khd->...hqk', q / scaling, k / scaling)
    if mask is not None:
      big_neg = jnp.finfo(attn_weights.dtype).min
      attn_weights = jnp.where(mask, attn_weights, big_neg)

    attn_weights = attn_weights.astype(f32)
    attn_weights = jax.nn.softmax(attn_weights).astype(dtype)
    out = jnp.einsum('...hqk,...khd->...qhd', attn_weights, v)
    out = out.reshape(*out.shape[:-2], -1)

    out = self.get('out', nets.Linear, inputs_q.shape[-1], **self._kw)(out)
    return out
