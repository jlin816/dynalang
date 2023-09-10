import collections
import pathlib
import sys
import time

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import embodied
import pytest

ALL = ['blocking', 'thread', 'process', 'daemon', 'process_slow']


class TestParallel:

  @pytest.mark.parametrize('parallel', ALL)
  def test_parallel_driver(self, parallel):
    env = embodied.envs.load_env(
        'dummy_discrete', parallel=parallel, amount=4, length=10)
    agent = embodied.RandomAgent(env.act_space)
    driver = embodied.Driver(env)
    episodes = collections.defaultdict(list)
    driver.on_episode(lambda ep, worker: episodes[worker].append(ep))
    start = time.time()
    driver(agent.policy, episodes=100)
    duration = time.time() - start
    print(parallel, duration)
    env.close()
