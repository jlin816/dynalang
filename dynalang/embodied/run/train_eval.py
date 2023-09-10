import re
from collections import defaultdict
from functools import partial as bind

import embodied
import numpy as np


def train_eval(
    agent, train_env, eval_env, train_replay, eval_replay, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)

  print_spaces = lambda title, spaces: [print(f'{title}:')] + [
      print(f'  {k:<10} {v}') for k, v in spaces if not k.startswith('log_')]
  print_spaces('Observation space', train_env.obs_space.items())
  print_spaces('Action space', train_env.act_space.items())

  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_eval = embodied.when.Every(args.eval_every, args.eval_initial)

  step = logger.step
  agg = embodied.Agg()
  epstats_train = embodied.Agg()
  epstats_eval = embodied.Agg()
  episodes_train = defaultdict(embodied.Agg)
  episodes_eval = defaultdict(embodied.Agg)
  nonzeros = set()

  def log_step(tran, worker, mode):

    episode = (episodes_eval if mode == 'eval' else episodes_train)[worker]
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')

    if tran['is_first']:
      episode.reset()

    for key, value in tran.items():
      if key in args.log_keys_video and worker < args.log_video_streams:
        episode.add(f'policy_{key}', tran[key], agg='stack')
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
      epstats = epstats_eval if mode == 'eval' else epstats_train
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length') - 1,
      }, prefix=f'episode_{mode}')
      rew = result.pop('rewards')
      result['reward_rate'] = (rew - rew.min() >= 0.1).mean()
      epstats.add(result)

  driver_train = embodied.Driver(train_env)
  driver_train.on_step(lambda tran, _: step.increment())
  driver_train.on_step(train_replay.add)
  driver_train.on_step(bind(log_step, mode='train'))

  driver_eval = embodied.Driver(eval_env)
  driver_eval.on_step(eval_replay.add)
  driver_eval.on_step(bind(log_step, mode='eval'))

  policy_train = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  policy_eval = lambda *args: agent.policy(*args, mode='eval')

  print('Prefill train dataset')
  while len(train_replay) < max(args.batch_steps, args.train_fill):
    driver_train(policy_train, steps=100)
  print('Prefill eval dataset')
  while len(eval_replay) < max(args.batch_steps, args.eval_fill):
    driver_eval(policy_eval, steps=100)

  dataset_train = agent.dataset(train_replay.dataset)
  dataset_eval = agent.dataset(eval_replay.dataset)
  state = [None]  # To be writable from train step function below.
  batch = [None]

  def train_step(tran, worker):
    for _ in range(should_train(step)):
      with embodied.timer.section('dataset_train_next'):
        batch[0] = next(dataset_train)
      outs, state[0], mets = agent.train(batch[0], state[0])
      agg.add(mets, prefix='train')
  driver_train.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we just saved.

  print('Start training loop')
  while step < args.steps:

    if should_eval(step):
      print('Starting evaluation at step', int(step))
      driver_eval.reset()
      driver_eval(policy_eval, episodes=args.eval_eps)

    driver_train(policy_train, steps=100)

    if should_log(step):
      with embodied.timer.section('dataset_eval_next'):
        eval_batch = next(dataset_eval)
      logger.add(agg.result())
      logger.add(epstats_eval.result(), prefix='epstats_eval')
      logger.add(epstats_train.result(), prefix='epstats_train')
      logger.add(agent.report(batch[0]), prefix='report_train')
      logger.add(agent.report(eval_batch), prefix='report_eval')
      logger.add(train_replay.stats(), prefix='train_replay')
      logger.add(eval_replay.stats(), prefix='eval_replay')
      logger.add(embodied.timer.stats(), prefix='timer')
      logger.write(fps=True)

    if should_save(step):
      checkpoint.save()

  logger.write()
  logger.write()
