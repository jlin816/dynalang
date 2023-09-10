import re
import sys
import time
from collections import defaultdict

import embodied
import numpy as np


def parallel(agent, replay, logger, make_env, num_envs, args):
  step = logger.step
  real_env_step = embodied.Counter()
  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('replay', replay, ['add', 'save'])
  timer.wrap('logger', logger, ['write'])
  usage = embodied.Usage(args.trace_malloc)
  workers = []
  if num_envs == 1:
    workers.append(embodied.distr.Thread(
        parallel_env, 0, make_env, args, timer))
  else:
    for i in range(num_envs):
      worker = embodied.distr.Process(parallel_env, i, make_env, args)
      worker.start()
      workers.append(worker)
    usage.processes('envs', workers)
  workers.append(embodied.distr.Thread(
      parallel_actor, step, real_env_step, agent, replay, logger, timer, args))
  workers.append(embodied.distr.Thread(
      parallel_learner, step, real_env_step, agent, replay, logger, timer, usage, args))
  embodied.distr.run(workers)


def parallel_actor(step, real_env_step, agent, replay, logger, timer, args):
  metrics = embodied.Metrics()
  scalars = defaultdict(lambda: defaultdict(list))
  videos = defaultdict(lambda: defaultdict(list))
  should_log = embodied.when.Clock(args.log_every)

  _, initial = agent.policy(dummy_data(
      agent.agent.obs_space, (args.actor_batch,)))
  initial = embodied.treemap(lambda x: x[0], initial)
  allstates = defaultdict(lambda: initial)
  nonzeros = set()
  vidstreams = {}
  dones = {}

  def callback(obs, env_addrs):
    metrics.scalar('parallel/ep_starts', obs['is_first'].sum(), agg='sum')
    metrics.scalar('parallel/ep_ends', obs['is_last'].sum(), agg='sum')
    for i, a in enumerate(env_addrs):
      if obs['is_first'][i]:
        abandoned = not dones.get(a, True)
        metrics.scalar('parallel/episode_abandoned', int(abandoned), agg='sum')
      dones[a] = obs['is_last'][i]

    states = [allstates[a] for a in env_addrs]
    states = embodied.treemap(lambda *xs: list(xs), *states)
    act, states = agent.policy(obs, states)
    act['reset'] = obs['is_last'].copy()
    for i, a in enumerate(env_addrs):
      allstates[a] = embodied.treemap(lambda x: x[i], states)
    step.increment(args.actor_batch)
    real_env_step.increment(
      (~obs["is_read_step"]).sum() if "is_read_step" in obs \
      else args.actor_batch
    )
    metrics.scalar('parallel/ep_states', len(allstates))

    trans = {**obs, **act}
    now = time.time()
    for i, a in enumerate(env_addrs):
      tran = {k: v[i].copy() for k, v in trans.items()}
      replay.add(tran.copy(), worker=a)  # Blocks when rate limited.
      if tran['is_first']:
        scalars.pop(a, None)
        videos.pop(a, None)
        vidstreams.pop(a, None)
      [scalars[a][k].append(v) for k, v in tran.items() if v.size == 1]
      if a in vidstreams or len(vidstreams) < args.log_video_streams:
        vidstreams[a] = now
        [videos[a][k].append(tran[k]) for k in args.log_keys_video]
    for a, last_add in list(vidstreams.items()):
      if now - last_add > args.log_video_timeout:
        print(f'Dropping video stream due to timeout ({now - last_add:.1f}s).')
        del vidstreams[a]
        del videos[a]

    for i, a in enumerate(env_addrs):
      if not trans['is_last'][i]:
        continue
      ep = scalars.pop(a)
      if a in vidstreams:
        ep.update(videos.pop(a))
        del vidstreams[a]
      ep = {k: embodied.convert(v) for k, v in ep.items()}
      logger.add({
          'real_length': len(ep['is_read_step']) - sum(ep['is_read_step']),
          'length': len(ep['reward']) - 1,
          'score': sum(ep['reward']),
      }, prefix='episode')
      logger.add({"real_step": real_env_step.value})
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

    if should_log():
      logger.add(metrics.result())

    return act

  server = embodied.distr.Server(
      callback, args.actor_port, args.ipv6, args.actor_batch,
      args.actor_threads)
  timer.wrap('server', server, ['_step', '_work'])
  server.run()


def parallel_learner(step, real_env_step, agent, replay, logger, timer, usage, args):
  logdir = embodied.Path(args.logdir)
  metrics = embodied.Metrics()
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.real_step = real_env_step 
  checkpoint.agent = agent
#  checkpoint.replay = replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()

  dataset = agent.dataset(replay.dataset)
  state = None
  stats = dict(last_time=time.time(), last_step=int(step), batch_entries=0)
  while True:

    batch = next(dataset)
    outs, state, mets = agent.train(batch, state)
    metrics.add(mets)
    stats['batch_entries'] += batch['is_first'].size

    if should_log():
      train = metrics.result()
      report = agent.report(batch)
      report = {k: v for k, v in report.items() if 'train/' + k not in train}
      logger.add(train, prefix='train')
      logger.add(report, prefix='report')
      logger.add(timer.stats(), prefix='timer')
      logger.add(replay.stats, prefix='replay')
      logger.add(usage.stats(), prefix='usage')

      duration = time.time() - stats['last_time']
      actor_fps = (int(step) - stats['last_step']) / duration
      learner_fps = stats['batch_entries'] / duration
      logger.add({
          'actor_fps': actor_fps,
          'learner_fps': learner_fps,
          'train_ratio': learner_fps / actor_fps if actor_fps else np.inf,
      }, prefix='parallel')
      stats = dict(last_time=time.time(), last_step=int(step), batch_entries=0)

      logger.write(fps=True)

    if should_save():
      checkpoint.save()


def parallel_env(replica_id, make_env, args, timer=None):
  # TODO: Optionally write NPZ episodes.
  assert replica_id >= 0, replica_id
  rid = replica_id
  print(f'[{rid}] Make env.')
  env = make_env()
  timer and timer.wrap('env', env, ['step'])
  addr = f'{args.actor_host}:{args.actor_port}'
  actor = embodied.distr.Client(addr, replica_id, args.ipv6)
  done = True
  start = time.time()
  count = 0
  while True:
    if done:
      act = {k: v.sample() for k, v in env.act_space.items()}
      act['reset'] = True
      score, length = 0, 0
    obs = env.step(act)
    obs = {k: np.asarray(v) for k, v in obs.items()}
    score += obs['reward']
    length += 1
    done = obs['is_last']
    if done:
      print(f'[{rid}] Episode of length {length} with score {score:.4f}.')
    promise = actor(obs)
    try:
      act = promise()
      act = {k: v for k, v in act.items() if not k.startswith('log_')}
    except embodied.distr.ReconnectError:
      print(f'[{rid}] Starting new episode because the client reconnected.')
      done = True
    except embodied.distr.RemoteError as e:
      print(f'[{rid}] Shutting down env due to agent error: {e}')
      sys.exit(0)
    count += 1
    now = time.time()
    if now - start >= 60:
      fps = count / (now - start)
      print(f'[{rid}] Env steps per second: {fps:.1f}')
      start = now
      count = 0


def dummy_data(spaces, batch_dims):
  # TODO: Get rid of this function by adding initial_policy_state() and
  # initial_train_state() to the agent API.
  spaces = list(spaces.items())
  data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces}
  for dim in reversed(batch_dims):
    data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
  return data
