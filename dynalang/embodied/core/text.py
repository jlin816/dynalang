import datasets
from transformers import T5Tokenizer
import numpy as np

class BatchedTextDataset:

  def __init__(self, name, batch_size, length, dataset_space, debug=False):
    if "_" in name:
      dataset = datasets.load_dataset(*name.split("_"))["train"]
    else:
      dataset = datasets.load_dataset(name)["train"]
    if debug:
      dataset = dataset.select(range(10000))
    dataset = dataset.shuffle(seed=42)
    tok = T5Tokenizer.from_pretrained("t5-small") 
    tokenize = lambda batch: tok(batch["text"])
    dataset = dataset.map(
      tokenize,
      batched=True,
      num_proc=4,
      remove_columns=dataset.column_names,
    )

    # Chunk the dataset, dropping remainders.
    def chunk_text(examples):
      keys = ["input_ids"]
      concat_examples = {k: sum(examples[k], []) for k in keys}
      total_length = len(concat_examples["input_ids"])
      total_length = (total_length // length) * length
      result = {
          k: [t[i : i + length] for i in range(0, total_length, length)]
          for k, t in concat_examples.items()
      }
#        result["labels"] = result["input_ids"].copy()
      result["token"] = np.array(result["input_ids"])
      return result
    dataset = dataset.map(
      chunk_text,
      batched=True,
      remove_columns=["attention_mask"],
    ).remove_columns(["input_ids"])

    dataset.set_format(type="numpy")

    self.batch_size = batch_size
    self.length = length
    self.epoch = 0
    self.dataset = dataset
    self.dataset_space = dataset_space
    self.shards = [self._shard(i) for i in range(self.batch_size)]

  def _shard(self, i):
    shard = iter(self.dataset.shard(num_shards=self.batch_size, index=i))
    zeros = {k: np.zeros((self.length, *space.shape)) for k, space in
             self.dataset_space.items() if not k.startswith("log_") and \
             k != "token"}
    while True:
      try:
        batch = next(shard)
      except StopIteration:
        if i == 0:
          self.epoch += 1
          print("One epoch done, re-shuffling.")
        shard = iter(dataset.shard(num_shards=self.batch_size, index=i).shuffle())
        batch = next(shard)
      batch.update(zeros)
      yield batch

  def __iter__(self):
    return self

  def __next__(self):
    batch = [next(shard) for shard in self.shards]
    batch = {k: np.stack([b[k] for b in batch], axis=0)
             for k, v in batch[0].items()}
    return batch
