import importlib
import os
import pathlib
import sys
import warnings
import wandb
from functools import partial as bind

# def warn_with_traceback(
#       message, category, filename, lineno, file=None, line=None):
#   log = file if hasattr(file, 'write') else sys.stderr
#   import traceback
#   traceback.print_stack(file=log)
#   log.write(warnings.formatwarning(
#       message, category, filename, lineno, line))
# warnings.showwarning = warn_with_traceback

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import embodied
from embodied import wrappers


def main(argv=None):

  embodied.print(r"---  ___                           __   ______ ---")
  embodied.print(r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---")
  embodied.print(r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---")
  embodied.print(r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---")

  from . import agent as agt
  parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
  config = embodied.Config(agt.Agent.configs['defaults'])
  for name in parsed.configs:
    config = config.update(agt.Agent.configs[name])
  config = embodied.Flags(config).parse(other)
  config = config.update({"logdir": f"{config.logdir}_{config.seed}"})
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.run.batch_size * config.batch_length)
  print('Run script:', args.script)

  logdir = embodied.Path(args.logdir)
  if args.script != 'parallel_env':
    logdir.mkdirs()
    config.save(logdir / 'config.yaml')
    step = embodied.Counter()
    logger = make_logger(parsed, logdir, step, config)

  embodied.timer.global_timer.enabled = args.timer

  cleanup = []
  try:

    if args.script == 'train':
      replay = make_replay(config, logdir / 'replay')
      env = wrapped_env(config, batch=True)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train(agent, env, replay, logger, args)

    elif args.script == 'train_save':
      replay = make_replay(config, logdir / 'replay')
      env = wrapped_env(config, batch=True)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_save(agent, env, replay, logger, args)

    elif args.script == 'train_eval':
      replay = make_replay(config, logdir / 'replay')
      eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
      env = wrapped_env(config, batch=True)
      eval_env = wrapped_env(config, batch=True)
      cleanup += [env, eval_env]
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_eval(
          agent, env, eval_env, replay, eval_replay, logger, args)

    elif args.script == 'train_holdout':
      replay = make_replay(config, logdir / 'replay')
      if config.eval_dir:
        assert not config.train.eval_fill
        eval_replay = make_replay(config, config.eval_dir, is_eval=True)
      else:
        assert 0 < args.eval_fill <= config.replay_size // 10, args.eval_fill
        eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
      env = wrapped_env(config, batch=True)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_holdout(
          agent, env, replay, eval_replay, logger, args)

    elif args.script == 'eval_only':
      env = wrapped_env(config, batch=True)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.eval_only(agent, env, logger, args)

    elif args.script == 'parallel':
      assert config.run.actor_batch <= config.envs.amount, (
          config.run.actor_batch, config.envs.amount)
      make_env2 = bind(wrapped_env, config, batch=False)
      step = embodied.Counter()
      env = make_env2()
      obs_space, act_space = env.obs_space, env.act_space
      env.close()
      agent = agt.Agent(obs_space, act_space, step, config)
      make_replay2 = bind(
          make_replay, config, logdir / 'replay', rate_limit=True)
      embodied.run.parallel(
          agent, logger, make_replay2, make_env2, config.envs.amount, args)

    elif args.script == 'parallel2':
      assert config.run.actor_batch <= config.envs.amount, (
          config.run.actor_batch, config.envs.amount)
      make_env2 = bind(wrapped_env, config, batch=False)
      step = embodied.Counter()
      env = make_env2()
      obs_space, act_space = env.obs_space, env.act_space
      env.close()
      agent = agt.Agent(obs_space, act_space, step, config)
      replay_path = str(logdir / 'replay')
      make_replay2 = bind(make_replay, config, replay_path, rate_limit=True)
      embodied.run.parallel2(
          agent, logger, make_replay2, make_env2, config.envs.amount, args)

    elif args.script == 'parallel3':
      assert config.run.actor_batch <= config.envs.amount, (
          config.run.actor_batch, config.envs.amount)
      make_env2 = bind(wrapped_env, config, batch=False)
      step = embodied.Counter()
      env = make_env2()
      obs_space, act_space = env.obs_space, env.act_space
      env.close()
      agent = agt.Agent(obs_space, act_space, step, config)
      replay_path = str(logdir / 'replay')
      make_replay2 = bind(make_replay, config, replay_path, rate_limit=True)
      logger_path = embodied.Path(str(logdir))
      make_logger2 = bind(make_logger, parsed, logger_path, None, config)
      embodied.run.parallel3(
          agent, make_logger2, make_replay2, make_env2, config.envs.amount,
          args)

    elif args.script == 'parallel_agent':
      make_env2 = bind(wrapped_env, config, batch=False)
      step = embodied.Counter()
      env = make_env2()
      obs_space, act_space = env.obs_space, env.act_space
      env.close()
      agent = agt.Agent(obs_space, act_space, step, config)
      replay_path = str(logdir / 'replay')
      make_replay2 = bind(make_replay, config, replay_path, rate_limit=True)
      logger_path = str(logdir)
      make_logger2 = bind(make_logger, parsed, logger_path, None, config)
      embodied.run.parallel3(
          agent, make_logger2, make_replay2, None, 0, args)

    elif args.script == 'parallel_env':
      ctor = bind(wrapped_env, config, batch=False)
      envid = args.env_replica
      if envid < 0:
        envid = int(os.environ['JOB_COMPLETION_INDEX'])
      embodied.run.parallel_env(envid, ctor, args)

    # elif args.script == 'train_with_text':
    #   replay = make_replay(config, logdir / 'replay')
    #   env = wrapped_env(config, batch=True)
    #   cleanup.append(env)
    #   step = embodied.Counter()
    #   agent = agt.Agent(env.obs_space, env.act_space, step, config)
    #   embodied.run.train_with_text(agent, env, replay, logger, args)

    else:
      raise NotImplementedError(args.script)
  finally:
    for obj in cleanup:
      obj.close()


def make_logger(parsed, logdir, step, config):
  if step is None:
    step = embodied.Counter()
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(config.filter, 'Agent'),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
      embodied.logger.TensorBoardOutput(
          logdir, config.run.log_video_fps, videos=config.tensorboard_videos),
  ], multiplier)
  if config.use_wandb:
    wandb_id_file = f"{str(logdir)}/wandb_id.txt"
    wandb_pa = embodied.Path(wandb_id_file)
    if wandb_pa.exists():
        print("!! Resuming wandb run !!")
        wandb_id = wandb_pa.read().strip()
    else:
        wandb_id = wandb.util.generate_id()
        wandb_pa.write(str(wandb_id))
    if "homegrid" in config.task:
      project = "homegridv3"
    elif "messenger" in config.task:
      project = "messenger"
    elif "homecook" in config.task:
      project = "homecook"
    elif "langroom" in config.task:
      project = "langroom"
    else:
      raise NotImplementedError
    wandb.init(
        id=wandb_id,
        resume="allow",
        project=project,
        name=logdir.name,
        group=logdir.name[:logdir.name.rfind("_")],
        sync_tensorboard=True,
        config=dict(config)
    )


  return logger


def make_replay(config, directory=None, is_eval=False, rate_limit=False):
  length = config.batch_length
  size = config.replay_size // 10 if is_eval else config.replay_size
  kwargs = {}
  if rate_limit and config.run.train_ratio > 0:
    kwargs['samples_per_insert'] = config.run.train_ratio / config.batch_length
    kwargs['tolerance'] = 10 * config.run.batch_size
    kwargs['min_size'] = config.run.batch_size
  replay = embodied.replay.Replay(length, size, directory, **kwargs)
  return replay


def wrapped_env(config, batch, **overrides):
  if batch:
    envs = []
    for index in range(config.envs.amount):
      ctor = bind(make_env, config, index, **overrides)
      if batch and config.envs.parallel != 'none':
        ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
      if config.envs.restarts:
        ctor = bind(wrappers.RestartOnException, ctor)
      envs.append(ctor())
    return embodied.BatchEnv(envs, config.envs.parallel)
  else:
    return make_env(config, index=0, **overrides)


def make_env(config, index, **overrides):
  from embodied.envs import from_gym
  suite, task = config.task.split('_', 1)
  if suite == 'procgen':  # TODO
    import procgen  # noqa
  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',
      'gym': 'embodied.envs.from_gym:FromGym',
      'dm': 'embodied.envs.from_dmenv:FromDM',
      'crafter': 'embodied.envs.crafter:Crafter',
      'dmc': 'embodied.envs.dmc:DMC',
      'atari': 'embodied.envs.atari:Atari',
      'atari100k': 'embodied.envs.atari:Atari',
      'dmlab': 'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav': 'embodied.envs.loconav:LocoNav',
      'pinpad': 'embodied.envs.pinpad:PinPad',
      'langroom': 'langroom:LangRoom',
      'procgen': lambda task, **kw: from_gym.FromGym(
          f'procgen:procgen-{task}-v0', **kw),  # TODO
  }[suite]
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)
  if kwargs.get('use_seed', False):
    kwargs['seed'] = hash((config.seed, index))
  env = ctor(task, **kwargs)
  return wrap_env(env, config)


def wrap_env(env, config):
  args = config.wrapper
  for name, space in env.act_space.items():
    if name == 'reset':
      continue
    elif not space.discrete:
      env = wrappers.NormalizeAction(env, name)
      if args.discretize:
        env = wrappers.DiscretizeAction(env, name, args.discretize)
  env = wrappers.ExpandScalars(env)
  if args.length:
    env = wrappers.TimeLimit(env, args.length, args.reset)
  if args.checks:
    env = wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = wrappers.ClipAction(env, name)
  return env


if __name__ == '__main__':
  main()
