import collections
import re
import warnings
import shutil

import embodied
from embodied.core import path

import numpy as np

def train(agent, env, replay, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_log_lang = embodied.when.Clock(5000)
  usage = embodied.Usage(args.trace_malloc)
  step = logger.step
  # Env steps without reading
  real_env_step = embodied.Counter()
  metrics = embodied.Metrics()
  print('Observation space:')
  for key, value in env.obs_space.items():
    print(f'  {key:<16} {value}')
  print('Action space:')
  for key, value in env.act_space.items():
    print(f'  {key:<16} {value}')

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', env, ['step'])
  timer.wrap('replay', replay, ['add', 'save'])
  timer.wrap('logger', logger, ['write'])

  nonzeros = set()
  def per_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    sum_abs_reward = float(np.abs(ep['reward']).astype(np.float64).sum())
    logger.add({
        'real_length': len(ep['is_read_step']) - sum(ep['is_read_step']),
        'length': length,
        'score': score,
        'sum_abs_reward': sum_abs_reward,
        'reward_rate': (np.abs(ep['reward']) >= 0.5).mean(),
    }, prefix='episode')
    logger.add({"real_step": real_env_step.value})
    print(f'Episode has {length} steps and return {score:.1f}.')
    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and np.all(value == 0):
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
    metrics.add(stats, prefix='stats')

  def count_real_step(tran):
    if not tran.get("is_read_step", False):
      real_env_step.increment()

  driver = embodied.Driver(env, exclude_keys=replay.dataset_excluded_keys)
  driver.on_episode(lambda ep, worker: per_episode(ep))
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: count_real_step(tran))
  driver.on_step(replay.add)

  random_agent = embodied.RandomAgent(env.act_space)
  print(f'Fill train dataset ({args.train_fill} steps).')
  while len(replay) < max(args.batch_steps, args.train_fill - len(replay)):
    driver(random_agent.policy, steps=100)
  logger.add(metrics.result())
  logger.write()

  dataset = agent.dataset(replay.dataset)
  state = [None]  # To be writable from train step function below.
  assert args.pretrain > 0  # At least one step to initialize variables.

  for pretrain_iter in range(args.pretrain):
    with timer.scope('dataset'):
      batch = next(dataset)
    _, state[0], mets = agent.train(batch, state[0])

  batch = [None]
  def train_step(tran, worker):
    for _ in range(should_train(step)):
      with timer.scope('dataset'):
        batch[0] = next(dataset)
      outs, state[0], mets = agent.train(batch[0], state[0])
      metrics.add(mets, prefix='train')
      if 'priority' in outs:
        replay.prioritize(outs['key'], outs['priority'])
    if should_log(step):
      agg = metrics.result()
      report = agent.report(batch[0])
      report = {k: v for k, v in report.items() if 'train/' + k not in agg}
      logger.add(agg)
      logger.add(report, prefix='report')
      logger.add(replay.stats, prefix='replay')
      logger.add(timer.stats(), prefix='timer')
      logger.add(usage.stats(), prefix='usage')
      logger.add({"real_step": real_env_step.value})
      logger.write(fps=True)
  driver.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  timer.wrap('checkpoint', checkpoint, ['save', 'load'])
  checkpoint.step = step
  checkpoint.real_step = real_env_step 
  checkpoint.agent = agent
# checkpoint.replay = replay
  if args.from_checkpoint:
#    shutil.copy(args.from_checkpoint, logdir / "init.ckpt")
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we jused saved.

  print('Start training loop.')
  policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  while step < args.steps:
    driver(policy, steps=100)
    if should_save(step):
      checkpoint.save()
