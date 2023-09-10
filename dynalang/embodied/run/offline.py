import collections
import re
import warnings

import embodied
import numpy as np
import json
import pickle
import shutil
import jax

TOKENIZER = None

def decode_tokens(tokens):
  global TOKENIZER
  if not TOKENIZER:
    from transformers import T5Tokenizer
    TOKENIZER = T5Tokenizer.from_pretrained("t5-small")
  if len(tokens.shape) > 2:
    tokens = tokens.reshape((-1, tokens.shape[-1]))
  return TOKENIZER.batch_decode(tokens) 

def offline(agent, offline_ds, eval_replay, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  step = logger.step
  metrics = embodied.Metrics()

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('logger', logger, ['write'])

  # Collect some random episodes on *train* env for monitoring loss.
#  print(f'Found {len(random_replay)} steps, filling.')
#  random_driver = embodied.Driver(env,
#    exclude_keys=random_replay.dataset_excluded_keys)
#  random_driver.on_step(random_replay.add)
#  random_agent = embodied.RandomAgent(env.act_space)
#  while len(random_replay) < max(args.batch_steps, args.train_fill):
#    random_driver(random_agent.policy, steps=100)
#  print(f"Finished filling random replay with {len(random_replay)} steps.")
#  random_dataset = agent.dataset(random_replay.dataset)
  eval_dataset = agent.dataset(eval_replay.dataset)

  dataset = iter(offline_ds)
  state = [None]  # To be writable from train step function below.

  # Pretraining mode: prepare to save checkpoint
  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.pkl')
  timer.wrap('checkpoint', checkpoint, ['save', 'load'])
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.load_or_save()
  should_save(step)
  print(f"Ckpt has step {checkpoint.step.value}")

  for pretrain_iter in range(args.pretrain):
    with timer.scope('dataset'):
      batch = next(dataset)
      batch = agent.postprocess(batch)
      if pretrain_iter == 0:
        print("Batch:")
        for k, v in batch.items():
          print(f"{k} {v.shape}")
    _, state[0], mets = agent.train(batch, state[0])
    # Count pretrain steps
    step.increment()
    metrics.add(mets, prefix="pretrain")
    if pretrain_iter % 500 == 0:
      agg = metrics.result()
      report = agent.report(batch)
      if "1step_openl_text" in report:
        report["1step_openl_text"] = decode_tokens(report["1step_openl_text"][:2])
        print("1 step:", report["1step_openl_text"])
      if "nstep_openl_text" in report:
        report["nstep_openl_text"] = decode_tokens(report["nstep_openl_text"][:2])
        print("n step:", report["nstep_openl_text"])
        print("True:", decode_tokens(jax.device_get(batch["token"])[:2]))
      report = {k: v for k, v in report.items() if 'pretrain/' + k not in agg}
      eval_report = agent.report(next(eval_dataset))
      logger.add(agg)
      logger.add(report, prefix='report')
      logger.add(eval_report, prefix='eval/report')
      logger.add(timer.stats(), prefix='timer')
      logger.add({"epoch": dataset.epoch})
      logger.write(fps=True)
    if should_save(pretrain_iter):
      checkpoint.save()
#      shutil.copy(logdir / 'checkpoint.pkl', logdir /
#                  f'checkpoint_{pretrain_iter}.pkl') 
        # Pickle a batch for inspection.
#        with open(logdir / f"batch.pkl", "wb") as f:
#          pickle.dump(batch, f)
  print('Pretraining done.')
  return
