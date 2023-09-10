import re

import embodied
import numpy as np

def save_frames(ep, outdir, img_key="image", text_key="token"):
  from PIL import Image
  import os
  os.makedirs(outdir, exist_ok=True)
  #  print(f"Saving ep with seq len {len(ep[img_key])}")
  for t, frame in enumerate(ep[img_key]):
    im = Image.fromarray(frame)
    im.save(f"{outdir}/{t}.png")
  with open(f"{outdir}/tokens.txt", "w") as f:
    for tok in ep[text_key]:
      f.write(f"{tok}\n")
  with open(f"{outdir}/rewards.txt", "w") as f:
    for tok in ep["reward"]:
      f.write(f"{tok}\n")
  with open(f"{outdir}/actions.txt", "w") as f:
    for tok in ep["action"]:
      f.write(f"{tok}\n")
  if "log_language_info" in ep:
    with open(f"{outdir}/lang.txt", "w") as f:
      for tok in ep["log_language_info"]:
        f.write(f"{tok}\n")


def eval_only(agent, env, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_log = embodied.when.Clock(args.log_every)
  step = logger.step
  metrics = embodied.Metrics()
  print('Observation space:', env.obs_space)
  print('Action space:', env.act_space)

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy'])
  timer.wrap('env', env, ['step'])
  timer.wrap('logger', logger, ['write'])

  nonzeros = set()
  fails = 0
  succs = 0
  thres = 3
  def per_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    logger.add({'length': length, 'score': score}, prefix='episode')
    print(f'Episode has {length} steps and return {score:.1f}.')
    nonlocal fails
    nonlocal succs
    if score <= thres:
      if fails < 3:
        path = f"{args.save_frames_to}/fail{fails}"
        fails += 1
        save_frames(ep, f"{path}", img_key="log_image")
#        save_frames(ep, f"{path}/fpv", img_key="image")
#        save_frames(ep, f"{path}/top", img_key="log_image")
    else:
      path = f"{args.save_frames_to}/succ{succs}"
      succs += 1
      save_frames(ep, f"{path}", img_key="log_image")
#      save_frames(ep, f"{path}/fpv", img_key="image")
#      save_frames(ep, f"{path}/top", img_key="log_image")
    if succs >= 4:
      exit()
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

  checkpoint = embodied.Checkpoint()
  checkpoint.agent = agent
  checkpoint.load(args.from_checkpoint, keys=['agent'])

  print('Start evaluation loop.')
  policy = lambda *args: agent.policy(*args, mode='eval')
  while step < args.steps:
    driver(policy, steps=100)
    if should_log(step):
      logger.add(metrics.result())
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  logger.write()
