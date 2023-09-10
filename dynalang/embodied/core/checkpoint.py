import concurrent.futures
import time

from . import basics
from . import path


class Checkpoint:

  def __init__(self, filename=None, parallel=True):
    self._filename = filename and path.Path(filename)
    self._values = {}
    self._parallel = parallel
    if self._parallel:
      self._worker = concurrent.futures.ThreadPoolExecutor(1)
      self._promise = None

  def __setattr__(self, name, value):
    if name in ('exists', 'save', 'load'):
      return super().__setattr__(name, value)
    if name.startswith('_'):
      return super().__setattr__(name, value)
    has_load = hasattr(value, 'load') and callable(value.load)
    has_save = hasattr(value, 'save') and callable(value.save)
    if not (has_load and has_save):
      message = f"Checkpoint entry '{name}' must implement save() and load()."
      raise ValueError(message)
    self._values[name] = value

  def __getattr__(self, name):
    if name.startswith('_'):
      raise AttributeError(name)
    try:
      return self._values.get(name)
    except AttributeError:
      raise ValueError(name)

  def exists(self, filename=None):
    assert self._filename or filename
    filename = path.Path(filename or self._filename)
    exists = self._filename.exists()
    if exists:
      print('Found existing checkpoint.')
    else:
      print('Did not find any checkpoint.')
    return exists

  def save(self, filename=None, keys=None):
    assert self._filename or filename
    filename = path.Path(filename or self._filename)
    print(f'Writing checkpoint: {filename}')
    if self._parallel:
      self._promise and self._promise.result()
      self._promise = self._worker.submit(self._save, filename, keys)
    else:
      self._save(filename, keys)

  def _save(self, filename, keys):
    keys = tuple(self._values.keys() if keys is None else keys)
    assert all([not k.startswith('_') for k in keys]), keys
    data = {k: self._values[k].save() for k in keys}
    data['_timestamp'] = time.time()
    filename.parent.mkdirs()
    # Write to a temporary file and then atomically rename, so that the
    # requested filename either contains a full checkpoint or does not exist if
    # writing was interrupted.
    tmp = filename.parent / (filename.name + '.tmp')
    tmp.write(basics.pack(data), mode='wb')
    tmp.move(filename)
    print(f'Wrote checkpoint: {filename}')

  def load(self, filename=None, keys=None):
    assert self._filename or filename
    self._promise and self._promise.result()  # Wait for last save.
    filename = path.Path(filename or self._filename)
    print(f'Loading checkpoint: {filename}')
    data = basics.unpack(filename.read('rb'))
    keys = tuple(data.keys() if keys is None else keys)
    for key in keys:
      if key.startswith('_'):
        continue
      if key not in self._values:
        print(f"Skipping {key} in checkpoint but not in agent.")
        continue
      try:
        self._values[key].load(data[key])
      except Exception:
        print(f'Error loading {key} from checkpoint.')
        raise
    age = time.time() - data['_timestamp']
    print(f'Loaded checkpoint from {age:.0f} seconds ago.')

  def load_or_save(self):
    if self.exists():
      self.load()
    else:
      self.save()
