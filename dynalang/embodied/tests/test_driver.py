import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import embodied
import numpy as np


class TestDriver:

  # Example trajectory:
  # idx: -1    0        1      ...  9      10      11
  # obs: zeros is_first mid         mid    is_last is_first
  # act: reset policy   policy      policy reset   policy

  def test_episode_length(self):
    env = embodied.envs.load_env('dummy_discrete', length=10)
    agent = embodied.RandomAgent(env.act_space)
    driver = embodied.Driver(env)
    seq = []
    driver.on_step(lambda tran, _: seq.append(tran))
    driver(agent.policy, episodes=1)
    assert len(seq) == 11

  def test_first_step(self):
    env = embodied.envs.load_env('dummy_discrete', length=10)
    agent = embodied.RandomAgent(env.act_space)
    driver = embodied.Driver(env)
    seq = []
    driver.on_step(lambda tran, _: seq.append(tran))
    driver(agent.policy, episodes=2)
    for index in [0, 11]:
      assert seq[index]['is_first'].item() is True
      assert seq[index]['is_last'].item() is False
    for index in [1, 10, 12]:
      assert seq[index]['is_first'].item() is False

  def test_last_step(self):
    env = embodied.envs.load_env('dummy_discrete', length=10)
    agent = embodied.RandomAgent(env.act_space)
    driver = embodied.Driver(env)
    seq = []
    driver.on_step(lambda tran, _: seq.append(tran))
    driver(agent.policy, episodes=2)
    for index in [10, 21]:
      assert seq[index]['is_last'].item() is True
      assert seq[index]['is_first'].item() is False
    for index in [0, 1, 9, 11, 20]:
      assert seq[index]['is_last'].item() is False

  def test_env_reset(self):
    env = embodied.envs.load_env('dummy_discrete', length=5)
    agent = embodied.RandomAgent(env.act_space)
    driver = embodied.Driver(env)
    seq = []
    driver.on_step(lambda tran, _: seq.append(tran))
    driver(lambda obs, state: ({'action': [[0, 1]]}, state), episodes=2)
    assert len(seq) == 12
    seq = {k: np.array([seq[i][k] for i in range(len(seq))]) for k in seq[0]}
    act = np.argmax(seq['action'], -1)
    assert (seq['is_first'] == [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).all()
    assert (seq['is_last']  == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]).all()
    assert (seq['reset']    == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]).all()
    assert (act             == [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]).all()

  def test_agent_inputs(self):
    env = embodied.envs.load_env('dummy_discrete', length=10)
    agent = embodied.RandomAgent(env.act_space)
    inputs = []
    states = []
    def policy(obs, state=None, mode='train'):
      inputs.append(obs)
      states.append(state)
      act, _ = agent.policy(obs, state, mode)
      return act, 'state'
    driver = embodied.Driver(env)
    seq = []
    driver.on_step(lambda tran, _: seq.append(tran))
    driver(policy, episodes=2)
    assert len(seq) == 22
    assert states == ([None] + ['state'] * 21)
    for index in [0, 11]:
      assert inputs[index]['is_first'].item() is True
    for index in [1, 10, 12, 21]:
      assert inputs[index]['is_first'].item() is False
    for index in [10, 21]:
      assert inputs[index]['is_last'].item() is True
    for index in [0, 1, 9, 11, 20]:
      assert inputs[index]['is_last'].item() is False

  def test_unexpected_reset(self):

    class UnexpectedReset(embodied.Wrapper):
      """Send is_first without preceeding is_last."""
      def __init__(self, env, when):
        super().__init__(env)
        self._when = when
        self._step = 0
      def step(self, action):
        if self._step == self._when:
          action = action.copy()
          action['reset'] = np.ones_like(action['reset'])
        self._step += 1
        return self.env.step(action)

    env = embodied.envs.load_env('dummy_discrete', length=4)
    env = UnexpectedReset(env, when=3)
    agent = embodied.RandomAgent(env.act_space)
    driver = embodied.Driver(env)
    steps = []
    episodes = []
    driver.on_step(lambda tran, _: steps.append(tran))
    driver.on_episode(lambda ep, _: episodes.append(ep))
    driver(agent.policy, episodes=1)
    assert len(steps) == 8
    steps = {k: np.array([x[k] for x in steps]) for k in steps[0]}
    assert (steps['reset'] == [0, 0, 0, 0, 0, 0, 0, 1]).all()
    assert (steps['is_first'] == [1, 0, 0, 1, 0, 0, 0, 0]).all()
    assert (steps['is_last'] == [0, 0, 0, 0, 0, 0, 0, 1]).all()
    assert len(episodes) == 1, 'only the second episode completed'
    assert (episodes[0]['reset'] == [0, 0, 0, 0, 1]).all()
    assert (episodes[0]['is_first'] == [1, 0, 0, 0, 0]).all()
    assert (episodes[0]['is_last'] == [0, 0, 0, 0, 1]).all()
