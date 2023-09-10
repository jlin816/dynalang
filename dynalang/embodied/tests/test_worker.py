import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import embodied
import pytest

ALL = ['blocking', 'thread', 'process', 'daemon', 'process_slow']


class TestWorker:

  @pytest.mark.parametrize('strategy', ALL)
  def test_consecutive_calls(self, strategy):
    worker = embodied.Worker(lambda x: x ** 2, strategy)
    for x in range(5):
      promise = worker(x=x)
      assert promise() == x ** 2
    worker.close()

  @pytest.mark.parametrize('strategy', ALL)
  def test_wait(self, strategy):
    worker = embodied.Worker(lambda x: x ** 2, strategy)
    for _ in range(2):
      promises = []
      for x in range(5):
        promises.append(worker(x))
      worker.wait()
      results = [p() for p in promises]
      assert results == [0, 1, 4, 9, 16]

  @pytest.mark.parametrize('strategy', ALL)
  def test_without_close(self, strategy):
    worker = embodied.Worker(lambda x: x ** 2, strategy)
    for x in range(5):
      promise = worker(x=x)
      assert promise() == x ** 2

  @pytest.mark.parametrize('strategy', ALL)
  def test_stateful_integer(self, strategy):
    def counter(state, x):
      print(state)
      state = state or 0
      state = state + 1
      return state, state
    worker = embodied.Worker(counter, strategy, state=True)
    promises = [worker(x) for x in range(6)]
    results = [promise() for promise in promises]
    assert results == [1, 2, 3, 4, 5, 6]

  @pytest.mark.parametrize('strategy', ALL)
  def test_stateful_dict(self, strategy):
    def triangle_number(state, x):
      state = state or {}
      current = x + state.get('last', 0)
      state['last'] = current
      return state, current
    worker = embodied.Worker(triangle_number, strategy, state=True)
    promises = [worker(x) for x in range(6)]
    results = [promise() for promise in promises]
    assert results == [0, 1, 3, 6, 10, 15]
