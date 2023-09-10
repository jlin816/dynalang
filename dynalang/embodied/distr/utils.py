import sys
import time
import traceback
import multiprocessing

import embodied

mp = multiprocessing.get_context('spawn')
PRINT_LOCK = mp.Lock()


def run(workers):

  for worker in workers:
    if not worker.started:
      worker.start()

  try:

    while True:

      if all(x.exitcode == 0 for x in workers):
        print('All workers terminated successfully.')
        return

      for worker in workers:
        if worker.exitcode not in (None, 0):
          print(f'Terminating worker due to error in {worker.name}')

          # Wait for everybody who wants to print their error messages.
          time.sleep(0.1)

          # Stop all workers that are not yet stopped.
          [x.terminate() for x in workers]

          msg = f'Terminated workers due to crash in {worker.name}.'
          raise RuntimeError(msg)
      time.sleep(0.1)

  finally:
    # Make sure all workers get stopped on shutdown. If some worker processes
    # survive program shutdown after an exception then ports may not be freeed
    # up. Even worse, clients of the new program execution could connect to
    # servers of the previous program execution that did not get cleaned up.
    [x.terminate() for x in workers]


def warn_remote_error(e, name, lock=PRINT_LOCK):
  summary = list(traceback.format_exception_only(e))[0].strip('\n')
  full = ''.join(traceback.format_exception(e)).strip('\n')
  msg = f"Exception in worker '{name}' ({summary}). "
  msg += 'Call check() to reraise in main process. '
  msg += f'Worker stack trace:\n{full}'
  with lock:
    embodied.print(msg, 'red')
  if sys.version_info.minor >= 11:
    e.add_note(f'\nWorker stack trace:\n\n{full}')


class Context:

  def __init__(self, predicate):
    self._predicate = predicate

  @property
  def running(self):
    return self._predicate()

  def __bool__(self):
    raise TypeError('Cannot convert Context to boolean.')
