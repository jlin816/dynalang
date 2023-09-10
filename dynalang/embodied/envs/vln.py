import embodied
import numpy as np
from VLN_CE.vlnce_baselines.config.default import get_config
from habitat_lab.habitat_baselines.utils.env_utils import make_env_fn
from habitat_lab.habitat_baselines.common.environments import get_env_class
import os
import random
from PIL import Image, ImageFont, ImageDraw
import pickle

class VLNEnv(embodied.Env):

  def __init__(
    self,
    task=None,
    mode='train',
    size=(64, 64),
    length=500,
    use_text=True,
    use_depth=False,
    load_embeddings=True,
    dataset='train',
    # For training with expert demos (unused in final version)
    use_expert=0,
    min_use_expert=0,
    anneal_expert_eps=0,
    # Reward for successful episode
    success_reward=1000,
    # Reward if STOP action executed too early
    early_stop_penalty=0,
    # Whether to include additional language beyond instruction in obs
    # (unused in final version)
    use_descriptions=False,
    desc_length=50,
    seed=None,
  ):
    assert mode in dataset, "Mismatched env mode and dataset"

    self._task = 'cont'
    self._size = size
    self._length = length
    self._step = 0
    self._done = False
    self._mode = mode
    self._use_text = use_text
    self._use_depth = use_depth
    self._load_embeddings = load_embeddings
    self._use_expert = use_expert
    self._use_descriptions = use_descriptions
    self._desc_length = desc_length
    self._min_use_expert = min_use_expert
    self._anneal_expert_eps = anneal_expert_eps
    self._success_reward = success_reward
    self._early_stop_penalty = early_stop_penalty
    # Reading timestep before start of episode
    self.read_step = 0
    # True if we have finished reading the first text input (the whole instr)
    self.done_first_input = False
    # Type of text we are currently inputting ('instr' or 'desc')
    self.cur_text_type = 'instr'
    # Text string currently being streamed
    self.cur_text = ''
    # Number of episodes (for annealing expert episodes if using demos)
    self._num_eps = 0

    if seed is None:
      seed = 42
    assert self._desc_length < self._length
    
    config_opts = [
      'TASK_CONFIG.DATASET.SPLIT', dataset,
      'TASK_CONFIG.TASK.NDTW.SPLIT', dataset,
      'TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE', mode == 'train',
      'TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE', mode != 'train',
      'TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE', mode == 'train',
    ]
    if mode == 'test':
      config_opts.extend([
        'TASK_CONFIG.TASK.SENSORS',
        ['INSTRUCTION_SENSOR']
      ])
      config_opts.extend([
        'TASK_CONFIG.TASK.MEASUREMENTS',
        ['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL', 'ORACLE_SUCCESS', 'NDTW', 'PATH_LENGTH']
      ])
    self.config = get_config(
      os.path.dirname(os.path.realpath(__file__)) + '/vln.yaml',
      opts=config_opts
    )
    self._env = make_env_fn(
      self.config,
      get_env_class(self.config.ENV_NAME)
    )

    if load_embeddings:
      with open(f"{os.path.dirname(__file__)}/data/vln_embeds_t5.pkl", "rb") as f:
        self.token_cache, self.embed_cache = pickle.load(f)
      self.empty_token_id = self.token_cache["<pad>"]
      self.empty_token_embed = self.embed_cache["<pad>"]
    else:
      self._init_models()

  def _init_models(self):
    """Initialize tokenizer and encoder for embedding online in the env."""
    self.token_cache = {}
    self.embed_cache = {}
    from transformers import T5Tokenizer, T5EncoderModel
    self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
    self.empty_token_id = self.tokenizer.pad_token_id
    self.encoder = T5EncoderModel.from_pretrained("t5-small")
    self.empty_token_embed = self._embed("<pad>")[0][0]    
 
  @property
  def obs_space(self):
    spaces = {k: embodied.Space(v.dtype, v.shape)
              for k, v in self._env.observation_space.items()}
    new_space = {}
    # resize image
    new_space['image'] = embodied.Space(
      dtype=spaces['rgb'].dtype,
      shape=self._size + (3,),
      low=np.zeros(self._size + (3,), dtype=np.int8),
      high=255 * np.ones(self._size + (3,), dtype=np.int8)
    )
    if self._use_depth:
      new_space['depth'] = new_space['image']

    if self._use_text:
      # use one field for instructions or description text
      new_space.update({
        "token": embodied.Space(
          low=0, high=32100,
          shape=(),
          dtype=np.uint32),
        "token_embed": embodied.Space(
          low=-np.inf, high=np.inf,
          shape=(512,),
          dtype=np.float32),
        "is_read_step": embodied.Space(
          low=np.array(False),
          high=np.array(True),
          shape=(),
          dtype=bool,
        )
      })

    new_space.update({
      f'log_{self._mode}_success': embodied.Space(np.float32),
      f'log_{self._mode}_pl_success': embodied.Space(np.float32),
      f'log_{self._mode}_oracle_success': embodied.Space(np.float32),
      f'log_image': new_space['image'],
      'reward': embodied.Space(np.float32),
      'is_last': embodied.Space(bool),
      'is_terminal': embodied.Space(bool),
      'is_first': embodied.Space(bool),
      'is_demo': embodied.Space(bool),
      'next_expert_ac': embodied.Space(np.int32, (), -1, len(self._disc_act_space))
    })
    return new_space

  @property
  def act_space(self):
    self._disc_act_space = ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']
    return {
        'action': embodied.Space(np.int32, (), 0, len(self._disc_act_space)),
        'reset': embodied.Space(bool),
    }
  
  def step(self, action):
    if self._done or action['reset']:
      self._num_eps += 1 
      self._step = 0
      self.read_step = 0
      self.cur_text = ''
      self.cur_text_type = 'instr'
      self.tokens = [] # for logging
      self._done = False
      self.done_first_input = False
      ob = self._env.reset()
      self.prev_env_ob = ob

      if self._num_eps < self._anneal_expert_eps:
        self._expert_ep = np.random.rand() < self._use_expert - (self._use_expert - self._min_use_expert) / self._anneal_expert_eps * self._num_eps
      elif self._min_use_expert == self._use_expert: 
        self._expert_ep = np.random.rand() < self._use_expert
      else:
        self._expert_ep = np.random.rand() < self._min_use_expert
 
      log_traj_id = ob['instruction']['trajectory_id']
      ob = self.preprocess_obs(ob)
      ob.update({
        'reward': 0,
        'is_first': True,
        'is_last': self._done,
        'is_terminal': False,
        f'log_{self._mode}_success': 0,
        f'log_{self._mode}_pl_success': 0,
        f'log_{self._mode}_oracle_success': 0,
        f'log_language_info': self.cur_text,
        "is_read_step": not self.done_first_input,
        "is_demo": self._expert_ep,
      })
      ob[f'log_image'] = self.render_with_text(
        ob, self.cur_text, log_traj_id, self._disc_act_space[action['action']]
      )

      if self._expert_ep:
        # need to get infos to get gt_actions
        self.next_expert_ac = self.prev_env_ob['shortest_path_sensor'][0]
        ob["next_expert_ac"] = self.next_expert_ac
      else:
        ob["next_expert_ac"] = -1
      
      return ob

    action = action['action'] # possible actions: STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT
    
    if self.done_first_input:
      self._step += 1
      ob, rew, dones, infos = self._env.step(action)
      if action == 0: # STOP
        if infos['success']: 
          rew = self._success_reward
        else:
          rew = self._early_stop_penalty # stop action too early
      self.prev_env_ob = ob
    else:
      # Agent needs to listen to instruction first, frozen
      ob = self.prev_env_ob
      rew = 0
      dones = False
      infos = {'success': 0, 'spl': 0, 'oracle_success': 0}
    
    log_traj_id = ob['instruction']['trajectory_id']
    ob = self.preprocess_obs(ob)
    if self._expert_ep:
      self.next_expert_ac = self.prev_env_ob['shortest_path_sensor'][0]
      ob["next_expert_ac"] = self.next_expert_ac
    else:
      ob["next_expert_ac"] = -1
    self._done = (self._step >= self._length) or dones
    ob.update({
      'reward': rew,
      'is_first': False,
      'is_last': (self._step >= self._length) or self._done,
      'is_terminal': self._done,
      "is_read_step": not self.done_first_input,
      "is_demo": self._expert_ep,
      f'log_{self._mode}_success': infos['success'],
      f'log_{self._mode}_pl_success': infos['spl'],
      f'log_{self._mode}_oracle_success': infos['oracle_success'],
      f'log_language_info': self.cur_text,
    })
    ob[f'log_image'] = self.render_with_text(
      ob, self.cur_text, log_traj_id, self._disc_act_space[action]
    )
    return ob

  def _embed(self, string):
    """Embed string with encoder or get from cache."""
    string = string.strip().replace('\n', ' ').replace('\r', '')
    
    if string not in self.embed_cache:
      print('Missing from cache!! String:', string)
      tokens = self.tokenizer(string, return_tensors="pt",
                              add_special_tokens=True)  # add </s> separators
      import torch
      with torch.no_grad():
        # (seq, dim)
        embeds = self.encoder(**tokens).last_hidden_state.squeeze(0)
      self.embed_cache[string] = embeds.cpu().numpy()
      self.token_cache[string] = {
        k: v.squeeze(0).cpu().numpy() for k, v in tokens}
    return (
      self.embed_cache[string],
      self.token_cache[string]
    )
   
  def get_embed_text(self, ob): 
    if len(self.tokens) > 0 and self.read_step >= len(self.tokens):
      self.read_step = 0
      self.done_first_input = True
      if self._use_descriptions  and len(ob['descriptions']) > 1 and self._step > 0:
        self.cur_text_type = 'instr' if self.cur_text_type == 'desc' else 'desc'
      else:
        self.cur_text_type = 'instr'

    if self.read_step == 0:
      # sample new text to feed in
      if self.cur_text_type == 'instr':
        self.cur_text = ob['instruction']['text']
      elif self.cur_text_type == 'desc':
        self.cur_text = random.choice(ob['descriptions'])
      else:
        raise NotImplementedError
      self.token_embeds = []
      self.tokens = [] # for logging

      # Remove padding 
      es, ts = self._embed(self.cur_text) # embed sentence
      self.token_embeds = [tok_e for tok_e in es]
      self.tokens = [tok for tok in ts]
      assert len(self.token_embeds) == len(self.tokens)

    # print(self.cur_text, self.read_step, self.tokens[self.read_step])
    new_ob = {
        "token": self.tokens[self.read_step],
        "token_embed": self.token_embeds[self.read_step],
      }
    self.read_step += 1
    return new_ob

  def preprocess_depth(self, depth):
    """Normalize and clip depth images."""
    depth = (np.clip(depth, 0, 5.0) / 5.0 * 255).astype(np.uint8) # Clip to 5m, convert to uint8
    depth = np.repeat(depth, 3, axis=-1)
    depth = Image.fromarray(depth)
    depth = depth.resize(self._size)
    depth = np.asarray(depth, dtype=np.uint8)
    return depth
      
  def preprocess_obs(self, ob):
    new_ob = {}
    img = Image.fromarray(ob['rgb'])
    img = img.resize(self._size)
    new_ob['image'] =  np.asarray(img, dtype=np.uint8)
    if self._use_depth:
      new_ob['depth'] = self.preprocess_depth(ob['depth'])
    if self._use_text:
      new_ob.update(self.get_embed_text(ob))
    return new_ob
  
  def render_with_text(self, ob, instr_text, traj_id, ac):
    """Render policy image with debugging information."""
    img = self._env.render()
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    # Define the maximum width of the text
    max_width = 256

    # Calculate the height of the text
    instr_text = 'Instruction: ' + instr_text
    instr_text = instr_text.encode("ascii", "ignore")
    instr_text = instr_text.decode()
    text_width, text_height = draw.textsize(instr_text)
    max_len = int((max_width / text_width) * len(instr_text))
    wrapped_text = "\n".join([instr_text[i:i+max_len] for i in range(0, len(instr_text), max_len)])

    draw.text((0, 0), 'Trajectory ID: {}, Mode: {}'.format(traj_id, self._mode), (0, 0, 0))
    draw.text((0, 15), "Action: {}".format(ac), (0, 0, 0))
    draw.multiline_text((0, 30), wrapped_text, fill=(0, 0, 0))
    img = np.asarray(img).copy()

    # annotate videos
    if ob[f'log_{self._mode}_success']:
      img[:5, :, 1] =  255
    if ob[f'log_{self._mode}_oracle_success']:
      img[:5, :, 2] =  255
    
    img = np.clip(img, 0, 255)
    return img
