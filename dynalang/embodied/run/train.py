import re
from collections import defaultdict

import embodied
import numpy as np


def train(agent, env, replay, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  step = logger.step

  print_spaces = lambda title, spaces: [print(f'{title}:')] + [
      print(f'  {k:<16} {v}') for k, v in spaces if not k.startswith('log_')]
  print_spaces('Observation space', env.obs_space.items())
  print_spaces('Action space', env.act_space.items())

  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)

  usage = embodied.Usage(**args.usage)
  agg = embodied.Agg()
  epstats = embodied.Agg()
  episodes = defaultdict(embodied.Agg)
  nonzeros = set()

  @embodied.timer.section('log_step')
  def log_step(tran, worker):

    episode = episodes[worker]
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')

    if tran['is_first']:
      episode.reset()

    if worker < args.log_video_streams:
      for key in args.log_keys_video:
        if key in tran:
          episode.add(f'policy_{key}', tran[key], agg='stack')
    for key, value in tran.items():
      if len(value.shape) > 0:
        continue
      if not args.log_zeros and key not in nonzeros and np.all(value == 0):
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        episode.add(key, value, agg='sum')
      if re.match(args.log_keys_avg, key):
        episode.add(key, value, agg='avg')
      if re.match(args.log_keys_max, key):
        episode.add(key, value, agg='max')

    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length') - 1,
      }, prefix='episode')
      rew = result.pop('rewards')
      result['reward_rate'] = (rew - rew.min() >= 0.1).mean()
      epstats.add(result)

  driver = embodied.Driver(env)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(replay.add)
  driver.on_step(log_step)

  policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  print('Prefill dataset')
  while len(replay) < max(args.batch_steps, args.train_fill):
    driver(policy, steps=100)

  dataset = agent.dataset(replay.dataset)
  state = [None]  # To be writable from train step function below.
  batch = [None]

  def train_step(tran, worker):
    for _ in range(should_train(step)):
      with embodied.timer.section('dataset_next'):
        batch[0] = next(dataset)
      outs, state[0], mets = agent.train(batch[0], state[0])
      agg.add(mets, prefix='train')
  driver.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.replay = replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we just saved.

  print('Start training loop')
  while step < args.steps:

    driver(policy, steps=10)

    if should_log(step):
      logger.add(agg.result(), prefix='train')
      logger.add(epstats.result(), prefix='epstats')
      logger.add(agent.report(batch[0]), prefix='report')
      logger.add(embodied.timer.stats(), prefix='timer')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.write(fps=True)

    if should_save(step):
      checkpoint.save()

  logger.write()
  logger.write()
