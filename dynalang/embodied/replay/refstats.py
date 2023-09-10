import gc
import types
from collections import defaultdict


class RefStats:

  def __init__(self, root, maxdepth=25):
    self.root = root
    self.maxdepth = maxdepth
    self.prev_objects = defaultdict(int)
    self.prev_chains = defaultdict(int)

  def __call__(self):
    objects, chains = self._traverse()
    delta_objects = {k: v - self.prev_objects[k] for k, v in objects.items()}
    delta_chains = {k: v - self.prev_chains[k] for k, v in chains.items()}
    self.prev_objects = objects
    self.prev_chains = chains
    abs_sorted = lambda d: sorted(d.items(), key=lambda x: -abs(x[1]))
    print('\nMost common new references:')
    for name, count in abs_sorted(delta_objects)[:50]:
      print(f'  {count:+d} ({objects[name]}) {name}')
    print('\nMost common new chains:')
    for chain, count in abs_sorted(delta_chains)[:50]:
      print(f'  {count:+d} ({chains[chain]}) ' + '/'.join(chain))

  def _traverse(self):
    objects = defaultdict(int)
    chains = defaultdict(int)
    visited = set()
    stack = [(self.root, (), ())]
    index = 0
    while stack:
      obj, path, parents = stack.pop()
      if 'RefStats' in path:
        continue
      if isinstance(obj, dict) and obj.get('__name__') == 'builtins':
        continue
      if isinstance(obj, types.ModuleType):
        continue
      index += 1
      if index % 10000 == 0:
        print(index)
      name = self._name(obj)
      for i in range(len(path)):
        chains[tuple(path[:i + 1] + (name,))] += 1
      if id(obj) in visited:
        continue
      visited.add(id(obj))
      objects[name] += 1
      if len(path) < self.maxdepth:
        try:
          refs = gc.get_referents(obj)
          stack.extend(
              (ref, path + (self._name(ref),), parents + (ref,))
              for ref in refs)
        except Exception:
          continue
    return objects, chains

  def _name(self, x):
    if not hasattr(x, '__name__'):
      x = type(x)
    x = x.__name__
    if not isinstance(x, str):
      x = repr(x)
    return x
