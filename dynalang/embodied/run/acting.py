import re
import time

import embodied
import numpy as np


def acting(agent, env, replay, logger, actordir, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir:', logdir)
  actordir = embodied.Path(actordir)
  actordir.mkdirs()
  should_expl = embodied.when.Until(args.expl_until)
  should_sync = embodied.when.Clock(args.sync_every)
  should_log = embodied.when.Clock(args.log_every)
  step = logger.step
  metrics = embodied.Metrics()
  print('Observation space:', env.obs_space)
  print('Action space:', env.act_space)

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy'])
  timer.wrap('env', env, ['step'])

  nonzeros = set()
  def per_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    logger.add({
        'length': length, 'score': score,
        'reward_rate': (ep['reward'] - ep['reward'].min() >= 0.1).mean(),
    }, prefix='episode')
    print(f'Episode has {length} steps and return {score:.1f}.')
    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
    metrics.add(stats, prefix='stats')

  driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: per_episode(ep))
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(replay.add)

  actor_cp = embodied.Checkpoint(actordir / 'actor.pkl')
  actor_cp.step = step
  actor_cp.load_or_save()

  fill = max(0, args.train_fill - int(step))
  if fill:
    print(f'Fill dataset ({fill} steps).')
    random_agent = embodied.RandomAgent(env.act_space)
    driver(random_agent.policy, steps=fill, episodes=1)

  # Initialize dataset and agent variables.
  agent.train(next(iter(agent.dataset(replay.dataset))))

  agent_cp = embodied.Checkpoint(logdir / 'agent.pkl')
  agent_cp.agent = agent

  print('Start collection loop.')
  policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')

  while step < args.steps:

    if should_log(step):
      logger.add(metrics.result())
      logger.write()

    if should_sync(step):
      print('Syncing.')
      actor_cp.save()
      while not agent_cp.exists():
        print('Waiting for agent checkpoint to be created.')
        time.sleep(10)
      for attempts in range(10):
        try:
          timestamp = agent_cp.load()
          if timestamp:
            logger.scalar('agent_cp_age', time.time() - timestamp)
          break
        except Exception as e:
          print(f'Could not load checkpoint: {e}')
        time.sleep(np.random.uniform(10, 60))
      else:
        raise RuntimeError('Failed to load checkpoint.')

    driver(policy, steps=100)
