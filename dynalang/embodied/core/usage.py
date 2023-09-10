from collections import defaultdict
import tracemalloc


class Usage:

  def __init__(self, trace_malloc=False):
    self.trace_malloc = trace_malloc
    if trace_malloc:
      tracemalloc.start()
      self._snapshot = None
    self.groups = {}

  def processes(self, name, procs):
    import psutil
    if not hasattr(procs, '__len__'):
      procs = [procs]
    procs = [int(x.pid if hasattr(x, 'pid') else x) for x in procs]
    procs = [psutil.Process(x) for x in procs]
    self.groups[name] = procs

  def stats(self):
    import psutil
    gb = 1024 ** 3
    cpus = psutil.cpu_count()
    memory = psutil.virtual_memory()
    stats = {
        'cpu_count': cpus,
        'cpu_frac': psutil.cpu_percent() / 100,
        'ram_total_gb': memory.total / gb,
        'ram_used_gb': memory.used / gb,
        'ram_avail_gb': memory.available / gb,
        'ram_frac': memory.percent / 100,
    }
    for name, group in self.groups.items():
      cpu = sum([x.cpu_percent() for x in group])
      stats[f'{name}_cpu_frac'] = cpu / cpus / 100
      ram = sum([x.memory_info().rss for x in group])
      stats[f'{name}_ram_gb'] = ram / gb
      stats[f'{name}_ram_frac'] = ram / memory.total
      stats[f'{name}_count'] = len(group)
    if self.trace_malloc:
      snapshot = tracemalloc.take_snapshot()
      stats['malloc_full'] = self._malloc_summary(snapshot)
      stats['malloc_diff'] = self._malloc_summary(snapshot, self._snapshot)
      self._snapshot = snapshot
      print(stats['malloc_full'])
    return stats

  def _malloc_summary(self, snapshot, relative=None, top=50, root='embodied'):
    if relative:
      statistics = snapshot.compare_to(relative, 'traceback')
    else:
      statistics = snapshot.statistics('traceback')
    agg = defaultdict(lambda: [0, 0])
    for stat in statistics:
      filename = stat.traceback[-1].filename
      lineno = stat.traceback[-1].lineno
      for frame in reversed(stat.traceback):
        if f'/{root}/' in frame.filename:
          filename = f'{root}/' + frame.filename.split(f'/{root}/')[-1]
          lineno = frame.lineno
          break
      agg[(filename, lineno)][0] += stat.size_diff if relative else stat.size
      agg[(filename, lineno)][1] += stat.count_diff if relative else stat.count
    lines = []
    lines.append('\nMemory Allocation' + (' Changes' if relative else ''))
    lines.append(f'\nTop {top} by size:\n')
    entries = sorted(agg.items(), key=lambda x: -abs(x[1][0]))
    for (filename, lineno), (size, count) in entries[:top]:
      size = size / (1024 ** 2)
      lines.append(f'- {size:.2f}Mb ({count}) {filename}:{lineno}')
    lines.append(f'\nTop {top} by count:\n')
    entries = sorted(agg.items(), key=lambda x: -abs(x[1][1]))
    for (filename, lineno), (size, count) in entries[:top]:
      size = size / (1024 ** 2)
      lines.append(f'- {size:.2f}Mb ({count}) {filename}:{lineno}')
    return '\n'.join(lines)
