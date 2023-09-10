import queue as queuelib
import sys
import threading
import traceback

import numpy as np


class Batcher:
  """Implements zip() with multi-threaded prefetching. The sources are expected to
  yield dicts of Numpy arrays and the iterator will return dicts of batched
  Numpy arrays."""

  def __init__(
      self, sources, workers=0, postprocess=None,
      prefetch_source=4, prefetch_batch=2,
      preprocessors=None):
    self._workers = workers
    self._postprocess = postprocess
    self.preprocessors = preprocessors or {}
    if workers:
      # Round-robin assign sources to workers.
      self._running = True
      self._threads = []
      self._queues = []
      assignments = [([], []) for _ in range(workers)]
      for index, source in enumerate(sources):
        queue = queuelib.Queue(prefetch_source)
        self._queues.append(queue)
        assignments[index % workers][0].append(source)
        assignments[index % workers][1].append(queue)
      for args in assignments:
        creator = threading.Thread(
            target=self._creator, args=args, daemon=True)
        creator.start()
        self._threads.append(creator)
      self._batches = queuelib.Queue(prefetch_batch)
      batcher = threading.Thread(
          target=self._batcher, args=(self._queues, self._batches,
                                      self.preprocessors),
          daemon=True)
      batcher.start()
      self._threads.append(batcher)
    else:
      self._iterators = [source() for source in sources]
      # Create all preprocessors in main thread.
      self.preprocessors = {k: v() for k, v in preprocessors.items()}
    self._once = False

  def close(self):
    if self._workers:
      self._running = False
      for thread in self._threads:
        thread.close()

  def __iter__(self):
    if self._once:
      raise RuntimeError(
          'You can only create one iterator per Batcher object to ensure that '
          'data is consumed in order. Create another Batcher object instead.')
    self._once = True
    return self

  def __call__(self):
    return self.__iter__()

  def __next__(self):
    if self._workers:
      batch = self._batches.get()
    else:
      elems = [next(x) for x in self._iterators]
      batch = {}
      for k in elems[0]:
        bx = [x[k] for x in elems]
        if k in self.preprocessors:
          preproc = self.preprocessors[k](bx)
          for preproc_key, preproc_val in preproc.items():
            batch[f"{k}_{preproc_key}"] = preproc_val
        else:
          batch[k] = np.stack(bx, 0)
      if self._postprocess:
        batch = self._postprocess(batch)
    if isinstance(batch, Exception):
      raise batch
    return batch

  def _creator(self, sources, outputs):
    try:
      iterators = [source() for source in sources]
      while self._running:
        for iterator, queue in zip(iterators, outputs):
          queue.put(next(iterator))
    except Exception as e:
      e.stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      outputs[0].put(e)
      raise

  def _batcher(self, sources, output, preprocessors):
    try:
      # Create preprocessors in-thread.
      preprocessors = {k: v() for k, v in preprocessors.items()}
      while self._running:
        elems = [x.get() for x in sources]
        for elem in elems:
          if isinstance(elem, Exception):
            raise elem
        batch = {}
        for k in elems[0]:
          bx = [x[k] for x in elems]
          if k in preprocessors:
            preproc = preprocessors[k](bx)
            for preproc_key, preproc_val in preproc.items():
              batch[f"{k}_{preproc_key}"] = preproc_val
          else:
            batch[k] = np.stack(bx, 0)
        if self._postprocess:
          batch = self._postprocess(batch)
        output.put(batch)  # Will wait here if the queue is full.
    except Exception as e:
      e.stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      output.put(e)
      raise
