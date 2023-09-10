import functools
import time

import numpy as np

from . import base
from . import space as spacelib

class OneHotObs(base.Wrapper):

  def __init__(self, env, key):
    super().__init__(env)
    self._count = int(env.obs_space[key].high)
    self._key = key

  @functools.cached_property
  def obs_space(self):
    shape = (self._count,)
    space = spacelib.Space(np.float32, shape, 0, 1)
    space.sample = functools.partial(self._sample, self._count)
    space._discrete = True
    return {**self.env.obs_space, self._key: space}

  def step(self, action):
    obs = self.env.step(action)
    one_hot = np.zeros(self._count)
    one_hot[obs[self._key]] = 1
    obs[self._key] = one_hot
    return obs

  @staticmethod
  def _sample(count):
    index = np.random.randint(0, count)
    action = np.zeros(count, dtype=np.float32)
    action[index] = 1.0
    return action



class ReadFirst(base.Wrapper):

  def __init__(self, env, duration, zero_obs=True):
    super().__init__(env)
    self._duration = duration
    self._zero_obs = zero_obs
    self._step = 0
    self._done = False
#    self._shape = self.env.obs_space['image'].shape
    self.init_obs = None
    print(f"ReadFirst: reading for {self._duration}, zero={self._zero_obs}")

  def step(self, action):
    if action['reset'] or self._done:
      self._step = 0
      self.init_obs = self.env.step(action)
    if self._step < self._duration:
      obs = self.init_obs
      if self._step > 0:
        obs['is_first'] = False
      if self._zero_obs:
        obs['image'] = np.zeros_like(self.init_obs['image'])
    else:
      obs = self.env.step(action)
    self._step += 1
    self._done = obs['is_last']
    return obs


class TimeLimit(base.Wrapper):

  def __init__(self, env, duration, reset=True, timeout_reward=0.0):
    super().__init__(env)
    self._duration = duration
    self._reset = reset
    self._step = 0
    self._done = False
    self.timeout_reward = timeout_reward

  def step(self, action):
    if action['reset'] or self._done:
      self._step = 0
      self._done = False
      if self._reset:
        action.update(reset=True)
        return self.env.step(action)
      else:
        action.update(reset=False)
        obs = self.env.step(action)
        obs['is_first'] = True
        return obs
    self._step += 1
    obs = self.env.step(action)
    if self._duration and self._step >= self._duration:
      obs['is_last'] = True
      obs['reward'] += self.timeout_reward
    self._done = obs['is_last']
    return obs


class ActionRepeat(base.Wrapper):

  def __init__(self, env, repeat):
    super().__init__(env)
    self._repeat = repeat
    self._done = False

  def step(self, action):
    if action['reset'] or self._done:
      return self.env.step(action)
    reward = 0.0
    for _ in range(self._repeat):
      obs = self.env.step(action)
      reward += obs['reward']
      if obs['is_last'] or obs['is_terminal']:
        break
    obs['reward'] = np.float32(reward)
    self._done = obs['is_last']
    return obs


class ClipAction(base.Wrapper):

  def __init__(self, env, key='action', low=-1, high=1):
    super().__init__(env)
    self._key = key
    self._low = low
    self._high = high

  def step(self, action):
    clipped = np.clip(action[self._key], self._low, self._high)
    return self.env.step({**action, self._key: clipped})


class NormalizeAction(base.Wrapper):

  def __init__(self, env, key='action'):
    super().__init__(env)
    self._key = key
    self._space = env.act_space[key]
    self._mask = np.isfinite(self._space.low) & np.isfinite(self._space.high)
    self._low = np.where(self._mask, self._space.low, -1)
    self._high = np.where(self._mask, self._space.high, 1)

  @functools.cached_property
  def act_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = spacelib.Space(np.float32, self._space.shape, low, high)
    return {**self.env.act_space, self._key: space}

  def step(self, action):
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self.env.step({**action, self._key: orig})


class OneHotAction(base.Wrapper):

  def __init__(self, env, key='action'):
    super().__init__(env)
    self._count = int(env.act_space[key].high)
    self._key = key

  @functools.cached_property
  def act_space(self):
    shape = (self._count,)
    space = spacelib.Space(np.float32, shape, 0, 1)
    space.sample = functools.partial(self._sample_action, self._count)
    space._discrete = True
    return {**self.env.act_space, self._key: space}

  def step(self, action):
    if not action['reset']:
      assert action[self._key].min() == 0.0, action
      assert action[self._key].max() == 1.0, action
      assert action[self._key].sum() == 1.0, action
    index = np.argmax(action[self._key])
    return self.env.step({**action, self._key: index})

  @staticmethod
  def _sample_action(count):
    index = np.random.randint(0, count)
    action = np.zeros(count, dtype=np.float32)
    action[index] = 1.0
    return action


class ExpandScalars(base.Wrapper):

  def __init__(self, env):
    super().__init__(env)
    self._obs_expanded = []
    self._obs_space = {}
    for key, space in self.env.obs_space.items():
      if space.shape == () and key != 'reward' and not space.discrete:
        space = spacelib.Space(space.dtype, (1,), space.low, space.high)
        self._obs_expanded.append(key)
      self._obs_space[key] = space
    self._act_expanded = []
    self._act_space = {}
    for key, space in self.env.act_space.items():
      if space.shape == () and not space.discrete:
        space = spacelib.Space(space.dtype, (1,), space.low, space.high)
        self._act_expanded.append(key)
      self._act_space[key] = space

  @functools.cached_property
  def obs_space(self):
    return self._obs_space

  @functools.cached_property
  def act_space(self):
    return self._act_space

  def step(self, action):
    action = {
        key: np.squeeze(value, 0) if key in self._act_expanded else value
        for key, value in action.items()}
    obs = self.env.step(action)
    obs = {
        key: np.expand_dims(value, 0) if key in self._obs_expanded else value
        for key, value in obs.items()}
    return obs


class FlattenTwoDimObs(base.Wrapper):

  def __init__(self, env):
    super().__init__(env)
    self._keys = []
    self._obs_space = {}
    for key, space in self.env.obs_space.items():
      if len(space.shape) == 2:
        space = spacelib.Space(
            space.dtype,
            (int(np.prod(space.shape)),),
            space.low.flatten(),
            space.high.flatten())
        self._keys.append(key)
      self._obs_space[key] = space

  @functools.cached_property
  def obs_space(self):
    return self._obs_space

  def step(self, action):
    obs = self.env.step(action).copy()
    for key in self._keys:
      obs[key] = obs[key].flatten()
    return obs


class CheckSpaces(base.Wrapper):

  def __init__(self, env):
    super().__init__(env)

  def step(self, action):
    for key, value in action.items():
      self._check(value, self.env.act_space[key], key)
    obs = self.env.step(action)
    for key, value in obs.items():
      if "language" in key or key.startswith("log_"): continue
      self._check(value, self.env.obs_space[key], key)
    return obs

  def _check(self, value, space, key):
    if not isinstance(value, (
        np.ndarray, np.generic, list, tuple, int, float, bool, str)):
      raise TypeError(f'Invalid type {type(value)} for key {key}.')
    if value in space:
      return
    dtype = np.array(value).dtype
    shape = np.array(value).shape
    lowest, highest = np.min(value), np.max(value)
    raise ValueError(
        f"Value for '{key}' with dtype {dtype}, shape {shape}, "
        f"lowest {lowest}, highest {highest} is not in {space}.")


class DiscretizeAction(base.Wrapper):

  def __init__(self, env, key='action', bins=5):
    super().__init__(env)
    self._dims = np.squeeze(env.act_space[key].shape, 0).item()
    self._values = np.linspace(-1, 1, bins)
    self._key = key

  @functools.cached_property
  def act_space(self):
    shape = (self._dims, len(self._values))
    space = spacelib.Space(np.float32, shape, 0, 1)
    space.sample = functools.partial(
        self._sample_action, self._dims, self._values)
    space._discrete = True
    return {**self.env.act_space, self._key: space}

  def step(self, action):
    if not action['reset']:
      assert (action[self._key].min(-1) == 0.0).all(), action
      assert (action[self._key].max(-1) == 1.0).all(), action
      assert (action[self._key].sum(-1) == 1.0).all(), action
    indices = np.argmax(action[self._key], axis=-1)
    continuous = np.take(self._values, indices)
    return self.env.step({**action, self._key: continuous})

  @staticmethod
  def _sample_action(dims, values):
    indices = np.random.randint(0, len(values), dims)
    action = np.zeros((dims, len(values)), dtype=np.float32)
    action[np.arange(dims), indices] = 1.0
    return action


class ResizeImage(base.Wrapper):

  def __init__(self, env, size=(64, 64)):
    super().__init__(env)
    self._size = size
    self._keys = [
        k for k, v in env.obs_space.items()
        if len(v.shape) > 1 and v.shape[:2] != size]
    print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
    if self._keys:
      from PIL import Image
      self._Image = Image

  @functools.cached_property
  def obs_space(self):
    spaces = self.env.obs_space
    for key in self._keys:
      shape = self._size + spaces[key].shape[2:]
      spaces[key] = spacelib.Space(np.uint8, shape)
    return spaces

  def step(self, action):
    obs = self.env.step(action)
    for key in self._keys:
      obs[key] = self._resize(obs[key])
    return obs

  def _resize(self, image):
    image = self._Image.fromarray(image)
    image = image.resize(self._size, self._Image.NEAREST)
    image = np.array(image)
    return image

class PadImage(base.Wrapper):

  def __init__(self, env, key, size=(16, 16)):
    super().__init__(env)
    self._size = size
    self._keys = [key]
    print(f'Resizing keys {",".join(self._keys)} to {self._size}.')

  @functools.cached_property
  def obs_space(self):
    spaces = self.env.obs_space
    for key in self._keys:
      shape = self._size + spaces[key].shape[2:]
      spaces[key] = spacelib.Space(np.uint8, shape)
    return spaces

  def step(self, action):
    obs = self.env.step(action)
    for key in self._keys:
      obs[key] = self._resize(obs[key])
    return obs

  def _resize(self, image):
    new = np.zeros((*self._size, image.shape[-1]))
    new[:image.shape[0], :image.shape[1]] = image
    return new


class RenderImage(base.Wrapper):

  def __init__(self, env, key='image'):
    super().__init__(env)
    self._key = key
    self._shape = self.env.render().shape

  @functools.cached_property
  def obs_space(self):
    spaces = self.env.obs_space
    spaces[self._key] = spacelib.Space(np.uint8, self._shape)
    return spaces

  def step(self, action):
    obs = self.env.step(action)
    obs[self._key] = self.env.render()
    return obs


class RestartOnException(base.Wrapper):

  def __init__(
      self, ctor, exceptions=(Exception,), window=300, maxfails=2, wait=20):
    if not isinstance(exceptions, (tuple, list)):
        exceptions = [exceptions]
    self._ctor = ctor
    self._exceptions = tuple(exceptions)
    self._window = window
    self._maxfails = maxfails
    self._wait = wait
    self._last = time.time()
    self._fails = 0
    super().__init__(self._ctor())

  def step(self, action):
    try:
      return self.env.step(action)
    except self._exceptions as e:
      if time.time() > self._last + self._window:
        self._last = time.time()
        self._fails = 1
      else:
        self._fails += 1
      if self._fails > self._maxfails:
        raise RuntimeError('The env crashed too many times.')
      message = f'Restarting env after crash with {type(e).__name__}: {e}'
      print(message, flush=True)
      time.sleep(self._wait)
      self.env = self._ctor()
      action['reset'] = np.ones_like(action['reset'])
      return self.env.step(action)


class StopAfterEpisodes(base.Wrapper):

  def __init__(self, env, limit, delay):
    super().__init__(env)
    self._limit = limit
    self._delay = delay
    self._episodes = 0
    self._ended = None

  def step(self, action):
    obs = self.env.step(action)
    if obs['is_last']:
      self._episodes += 1
    if self._episodes >= self._limit:
      if self._ended is None:
        self._ended = time.time()
      elif time.time() - self._ended > self._delay:
        print('Finished experiment.')
        from google3.learning.deepmind.xmanager2.client import xmanager_api as xm
        xm.XManagerApi().get_current_work_unit().stop()
        while True:
          time.sleep(5)
    return obs


class RewardScale(base.Wrapper):

  def __init__(self, env, scale=1.0):
    super().__init__(env)
    self._scale = scale

  def step(self, action):
    obs = self.env.step(action)
    obs['reward'] *= self._scale
    return obs
