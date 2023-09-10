import concurrent.futures
from collections import defaultdict, deque
from functools import partial as bind

import embodied

from . import chunk as chunklib


class Saver:

  def __init__(self, directory, chunks=1024):
    self.directory = embodied.Path(directory)
    self.directory.mkdirs()
    self.chunks = chunks
    self.buffers = defaultdict(bind(chunklib.Chunk, chunks))
    self.workers = concurrent.futures.ThreadPoolExecutor(16)
    self.promises = deque()
    self.loading = False

  def add(self, step, worker):
    if self.loading:
      return
    buffer = self.buffers[worker]
    buffer.append(step)
    if buffer.length >= self.chunks:
      self.buffers[worker] = buffer.succ = chunklib.Chunk(self.chunks)
      self.promises.append(self.workers.submit(buffer.save, self.directory))
      for promise in [x for x in self.promises if x.done()]:
        promise.result()
        self.promises.remove(promise)

  def save(self, wait=False):
    for buffer in self.buffers.copy().values():
      if buffer.length:
        self.promises.append(self.workers.submit(buffer.save, self.directory))
    if wait:
      [x.result() for x in self.promises]
      self.promises.clear()

  def load(self, capacity, length):
    filenames = scan(self.directory, capacity, length - 1)
    if not filenames:
      return
    threads = min(len(filenames), 32)
    promises = []
    chunks = []
    with concurrent.futures.ThreadPoolExecutor(threads) as executor:
      for filename in filenames:
        promises.append(executor.submit(chunklib.Chunk.load, filename))
      for filename, promise in zip(filenames, promises):
        try:
          chunks.append(promise.result())
        except Exception as e:
          print(f'Error loading chunk {filename}: {e}')
    streamids = {}
    for chunk in reversed(sorted(chunks, key=lambda x: x.time)):
      if chunk.succ not in streamids:
        streamids[chunk.uuid] = int(embodied.uuid())
      else:
        streamids[chunk.uuid] = streamids[chunk.succ]
    self.loading = True
    for i, chunk in enumerate(chunks):
      stream = streamids[chunk.uuid]
      for index in range(chunk.length):
        step = {k: v[index] for k, v in chunk.data.items()}
        yield step, stream
      # Free memory early to not require twice the replay capacity.
      chunks[i] = None
      del chunk
    self.loading = False


def scan(directory, capacity=None, shorten=0):
  directory = embodied.Path(directory)
  filenames, total = [], 0
  for filename in reversed(sorted(directory.glob('*.npz'))):
    if capacity and total >= capacity:
      break
    filenames.append(filename)
    total += max(0, int(filename.stem.split('-')[3]) - shorten)
  return sorted(filenames)
