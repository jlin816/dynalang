import embodied
import numpy as np
import queue 
import threading
import pickle

class OfflineDataset:

  def __init__(self, batch, length, directories, keys=["image", "is_first"], data_loaders=None,
               prefetch=1, chunk_capacity=None, seed=0, postprocess=None):
    self.batch = batch
    self.length = length
    self.data_loaders = data_loaders if data_loaders else batch
    self.chunk_length = 1024
    self.seed = seed
    self.prefetch = prefetch
    self.postprocess = postprocess if postprocess else lambda x: x 
    self.epoch = 0
    self.keys = keys

    assert self.data_loaders <= batch, f"# data loaders must be <= batch."
    assert self.length < self.chunk_length
    self.directories = [embodied.Path(d) for d in directories]
    self.chunks = [
      c for d in self.directories for c in d.glob("*pkl")
#      if int(c.stem.split('-')[3]) >= self.chunk_length
    ]
    assert len(self.chunks) > 0, f"empty directory {directories}"
    if chunk_capacity and len(self.chunks) > chunk_capacity:
      print(f"Filling to max capacity {chunk_capacity}.")
      self.chunks = self.chunks[:int(chunk_capacity)]
    print("Total # offline chunks: ", len(self.chunks))
    self.workers = [
      threading.Thread(
        target=self._worker,
        args=(i,),
        daemon=True,
      ) for i in range(self.data_loaders)
    ]
    self.output_qs = [queue.Queue(prefetch) for _ in range(self.data_loaders)]
    self.batched_q = queue.Queue(prefetch)
    self.batcher = threading.Thread(
      target=self._batcher, 
      daemon=True
    )
    print("Starting dataloader workers.")
    for t in self.workers + [self.batcher]:
      t.start()

  def __iter__(self):
    return self

  def __next__(self):
    return self.batched_q.get()

  def _worker(self, i):
    epoch = 0
    data = np.copy(self.chunks)
    np.random.seed(self.seed)
    while True:
      np.random.shuffle(data)
      for fname in data[i::self.batch]:
        start = np.random.randint(0, self.chunk_length - self.length)
        with fname.open("rb") as f:
          x = pickle.load(f)
          x = {k: v[start:start + self.length] for k, v in x.items()}
        assert "token" not in x or len(x["token"].shape) == 1
        self.output_qs[i].put(x)
      #print(f"[w{i}] Epoch {epoch} done")
      epoch += 1
      if i == 0:
        self.epoch = epoch

  def _batcher(self):
    while True:
      batch = [self.output_qs[i % self.data_loaders].get() for i in
               range(self.batch)] 
      batch = {k: np.stack([x[k] for x in batch], 0) for k in self.keys}
      batch = self.postprocess(batch)
      self.batched_q.put(batch)

if __name__ == "__main__":
  ds = iter(OfflineDataset(5, 64, []))
  for _ in range(10):
    print(f"Batch: {next(ds)}")
