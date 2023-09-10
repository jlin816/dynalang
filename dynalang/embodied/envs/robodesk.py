import os

import embodied

from . import gym


class RoboDesk(embodied.Env):

  def __init__(self, task, mode, repeat=1, length=500, resets=True):
    assert mode in ('train', 'eval')
    # TODO: This env variable is necessary when running on a headless GPU but
    # breaks when running on a CPU machine.
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
    try:
      from robodesk import robodesk
    except ImportError:
      import robodesk
    task, reward = task.rsplit('_', 1)
    if mode == 'eval':
      reward = 'success'
    assert reward in ('dense', 'sparse', 'success'), reward
    self._gymenv = robodesk.RoboDesk(task, reward, repeat, length)
    self._env = gym.Gym(self._gymenv)

  @property
  def obs_space(self):
    return self._env.obs_space

  @property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    obs = self._env.step(action)
    obs['is_terminal'] = False
    return obs
