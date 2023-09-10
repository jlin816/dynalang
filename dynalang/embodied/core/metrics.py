import collections
import warnings

import numpy as np


class Metrics:

  def __init__(self):
    self.scalars = collections.defaultdict(list)
    self.aggs = {}
    self.lasts = {}

  def scalar(self, key, value, agg='mean'):
    assert agg in ('mean', 'sum', 'min', 'max')
    self.scalars[key].append(value)
    self.aggs[key] = agg

  def image(self, key, value):
    self.lasts[key] = value

  def video(self, key, value):
    self.lasts[key] = value

  def add(self, mapping, prefix=None):
    for key, value in mapping.items():
      key = prefix + '/' + key if prefix else key
      if hasattr(value, 'shape') and len(value.shape) > 0:
        self.lasts[key] = value
      else:
        self.scalar(key, value)

  def result(self, reset=True):
    result = {}
    result.update(self.lasts)
    with warnings.catch_warnings():  # Ignore empty slice warnings.
      warnings.simplefilter('ignore', category=RuntimeWarning)
      for key, values in self.scalars.items():
        agg = self.aggs[key]
        value = {
            'mean': np.nanmean,
            'sum': np.nansum,
            'min': np.nanmin,
            'max': np.nanmax,
        }[agg](values, dtype=np.float64)
        result[key] = value
    reset and self.reset()
    return result

  def reset(self):
    self.scalars.clear()
    self.lasts.clear()
