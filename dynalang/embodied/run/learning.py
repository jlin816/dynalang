import collections
import time
import warnings

import embodied
import numpy as np


def learning(agent, train_replay, eval_replay, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_sync = embodied.when.Clock(args.sync_every)
  should_log = embodied.when.Clock(args.log_every)
  should_eval = embodied.when.Clock(args.eval_every)
  step = logger.step
  metrics = embodied.Metrics()

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['train', 'report', 'save'])
  if hasattr(train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])

  dataset_train = iter(agent.dataset(train_replay.dataset))
  dataset_eval = None  # Initialize later to reduce startup time.
  _, state, _ = agent.train(next(dataset_train))

  agent_cp = embodied.Checkpoint(logdir / 'agent.pkl')
  agent_cp.agent = agent
  agent_cp.load_or_save()

  learner_cp = embodied.Checkpoint(logdir / 'learner.pkl')
  learner_cp.train_replay = train_replay
  learner_cp.step = step
  learner_cp.load_or_save()

  # Wait for prefill data from at least one actor to avoid overfitting to only
  # small amount of data that is read first.
  while len(train_replay) < args.train_fill:
    print('Waiting for train data prefill...')
    time.sleep(10)

  print('Start training loop.')
  while step < args.steps:

    batch = next(dataset_train)
    outs, state, mets = agent.train(batch, state)
    metrics.add(mets, prefix='train')
    if 'priority' in outs:
      train_replay.prioritize(outs['key'], outs['priority'])
    step.increment()

    if should_log(step):
      logger.add(metrics.result())
      logger.add(agent.report(batch), prefix='report')
      if dataset_eval:
        if not dataset_eval:
          print('Initializing eval replay...')
          dataset_eval = iter(agent.dataset(eval_replay.dataset))
        logger.add(agent.report(next(dataset_eval)), prefix='eval')
      logger.add(train_replay.stats, prefix='replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)

    if should_sync(step):
      agent_cp.save()
      learner_cp.save()
