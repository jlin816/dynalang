import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import embodied
import numpy as np
import pytest


class TestRunParallel:

  def test_replay_order(self):
    args = self.make_args()
    env = self.make_env()
    agent = DummyAgent(env.obs_space, env.act_space)
    env.close()
    replay = embodied.replay.Replay(length, size)
    logger = embodied.Logger(embodied.Counter(), [])
    embodied.run.parallel(
        agent, replay, logger, self.make_env, num_envs=5, args=args)

  def make_env(self):
    from embodied.envs import dummy
    return dummy.Dummy('disc', size=(64, 64), length=100)

  def make_args(self):
    return embodied.Config(
        steps=1e10,
        # expl_until=0,
        log_every=120,
        save_every=900,
        # eval_every=1e6,
        # eval_initial=True,
        # eval_eps=1,
        # eval_samples=1,
        train_ratio=32.0,
        # train_fill=0,
        # eval_fill=0,
        log_zeros=False,
        log_keys_video=['image'],
        log_keys_sum='^$',
        log_keys_avg='^$',
        log_keys_max='^$',
        log_video_fps=20,
        log_video_streams=4,
        log_episode_timeout=60,
        from_checkpoint='',
        actor_host='localhost',
        actor_port='5551',
        actor_batch=32,
        actor_threads=1,
        # env_replica=-1,
        ipv6=False,
        usage=dict(psutil=False, gputil=False, malloc=False, gc=False),
        env_processes=True,
        enable_timer=True,
    )


class DummyAgent:

  def __init__(self, obs_space, act_space):
    self.obs_space = obs_space
    self.act_space = act_space

  def policy(self, obs, carry=(), mode='train'):
    batch_size = len(obs['is_first'])
    act = {
        k: np.stack([v.sample() for _ in range(batch_size)])
        for k, v in self.act_space.items() if k != 'reset'}
    return act, carry

  def train(self, data, carry=()):
    B, T = data['step'].shape
    actual = data['step'] - data['step'][:, :1]
    reference = np.repeat(np.arange(T)[None], B, 0)
    assert (actual == reference).all()
    outs = {}
    metrics = {}
    return outs, carry, metrics

  def report(self, data):
    report = {}
    return report

  def dataset(self, generator):
    return generator()

  def save(self):
    return None

  def load(self, data=None):
    pass
