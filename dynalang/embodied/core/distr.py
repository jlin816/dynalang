import concurrent.futures
import ctypes
import sys
import threading
import time
import traceback
from collections import deque

import numpy as np

from . import basics


class RemoteError(RuntimeError): pass
class ReconnectError(RuntimeError): pass


class Client:

  def __init__(self, address, identity=None, ipv6=False, timeout=10):
    if identity is None:
      identity = np.random.randint(2 ** 32)
    assert isinstance(identity, int), (type(identity), identity)
    self.address = address
    self.identity = identity
    self.ipv6 = ipv6
    self.timeout = timeout
    self.socket = None
    self.pending = False
    self.once = True
    self._connect()

  def __call__(self, data):
    assert isinstance(data, dict), type(data)
    if self.pending:
      self._receive()
    self.socket.send(basics.pack(data))
    self.once and self._print('Sent first request.')
    self.pending = True
    return self._receive

  def _connect(self):
    import zmq
    context = zmq.Context.instance()
    self.socket = context.socket(zmq.REQ)
    self.socket.setsockopt(zmq.IDENTITY, self.identity.to_bytes(16, 'big'))
    self.ipv6 and self.socket.setsockopt(zmq.IPV6, 1)
    self.socket.RCVTIMEO = int(1000 * self.timeout)
    address = self._resolve(self.address)
    self._print(f'Client connecting to {address}', color='green')
    self.socket.connect(address)
    self.pending = False
    self.once = True

  def _resolve(self, address):
    return f'tcp://{address}'

  def _receive(self):
    import zmq
    try:
      while True:
        received = self.socket.recv()
        self.once and self._print('Received first response.')
        self.once = False
        if received == b'wait':
          self.socket.send(b'waiting')
          continue
        else:
          break
    except zmq.Again:
      self._print('Reconnecting because server did not respond.', color='red')
      self.socket.close(linger=0)
      self._connect()
      raise ReconnectError()
    result = basics.unpack(received)
    if result.get('type', 'data') == 'error':
      msg = result.get('message', None)
      raise RemoteError(f'Server responded with an error: {msg}')
    self.pending = False
    return result

  def _print(self, text, color=None):
    text = f'[{self.identity}] {text}'
    if color:
      basics.print_(text, color=color)
    else:
      print(text)


class Server:

  def __init__(self, function, port, ipv6=False, batch=-1, threads=1):
    import zmq
    context = zmq.Context.instance()
    self.socket = context.socket(zmq.ROUTER)
    ipv6 and self.socket.setsockopt(zmq.IPV6, 1)
    address = f'tcp://*:{port}'
    basics.print_(f'BatchServer listening at {address}', color='green')
    self.socket.bind(address)
    self.function = function
    self.batch = batch
    self.workers = concurrent.futures.ThreadPoolExecutor(threads)
    self.promises = deque()
    self.inputs = deque()
    self.requests = {}
    self.outputs = {}
    self.once = True
    self.error = None

  def run(self):
    while True:
      start = time.time()
      self._step()
      duration = time.time() - start
      time.sleep(max(0, 0.001 - duration))

  def _step(self):
    import zmq

    # If there are new messages, dispatch them to the respective queue.
    try:
      while True:
        now = time.time()
        addr, empty, message = self.socket.recv_multipart(zmq.NOBLOCK)
        self.requests[addr] = now
        if message != b'waiting':
          self.inputs.append((addr, message))
    except zmq.Again:
      pass

    # If we have accumulated enough inputs, remove them from the queue and
    # dispatch them to the worker pool.
    if len(self.inputs) >= max(1, self.batch):
      inputs = [self.inputs.popleft() for _ in range(max(1, self.batch))]
      addrs, inputs = [a for a, x in inputs], [x for a, x in inputs]
      self.promises.append(self.workers.submit(self._work, addrs, inputs))

    # If any background tasks have finished, remove their promises from the
    # queue and resolve them.
    while self.promises and self.promises[0].done():
      self.promises.popleft().result()

    # If any of the background tasks have set the error field, then wait for
    # all other background tasks to finish first.
    if self.error:
      [x.result() for x in self.promises]

    # Send all available results back to their respective clients. The
    # result is sent instead of the heartbeat, so we remove the heartbeat for
    # the same client from the queue.
    for addr in list(self.outputs.keys()):
      if addr not in self.requests:
        # This can happen if we just sent a heartbeat recently and have not
        # received confirmation from the client yet.
        continue
      message = self.outputs.pop(addr)
      del self.requests[addr]
      # When ROUTER sockets reply to clients that are unreachable, they drop
      # messages by default, which is what we want here.
      # https://zguide.zeromq.org/docs/chapter3/#ROUTER-Error-Handling
      self.socket.send_multipart([addr, b'', message])
      # self.socket.send_multipart([addr, b'', message], zmq.NOBLOCK)

    # Respond with waiting to requests that are older than one second.
    now = time.time()
    for addr, arrival in list(self.requests.items()):
      if now - arrival >= 1.0:
        del self.requests[addr]
        self.socket.send_multipart([addr, b'', b'wait'])
        # self.socket.send_multipart([addr, b'', message], zmq.NOBLOCK)

    # If any of the background tasks have set the error field, raise the
    # error here after we have responded to the clients.
    if self.error:
      raise self.error

  def _work(self, addrs, inputs):
    error = None
    inputs = [basics.unpack(x) for x in inputs]
    inputs = {
        k: [inputs[i][k] for i in range(len(inputs))]
        for k in inputs[0].keys()}
    inputs = {
        k: v if isinstance(v[0], str) else np.asarray(v)
        for k, v in inputs.items()}
    if self.batch < 1:
      inputs = {k: v[0] for k, v in inputs.items()}
    try:
      results = self.function(inputs, [x.hex() for x in addrs])
      if self.batch <= 0:
        results = {k: [v] for k, v in results.items()}
      results = {
          a: {k: v[i] for k, v in results.items()}
          for i, a in enumerate(addrs)}
    except Exception as e:
      error = e
      results = {a: {'type': 'error', 'message': str(e)} for a in addrs}
    results = {a: basics.pack(v) for a, v in results.items()}
    self.outputs.update(results)
    if error and not self.error:
      self.error = error


class Thread(threading.Thread):

  lock = threading.Lock()

  def __init__(self, fn, *args, name=None):
    self.fn = fn
    self.exitcode = None
    name = name or fn.__name__
    super().__init__(target=self._wrapper, args=args, name=name, daemon=True)

  @property
  def running(self):
    return self.is_alive()

  def _wrapper(self, *args):
    try:
      self.fn(*args)
    except Exception:
      with self.lock:
        print('-' * 79)
        print(f'Exception in worker: {self.name}')
        print('-' * 79)
        print(''.join(traceback.format_exception(*sys.exc_info())))
        self.exitcode = 1
      raise
    self.exitcode = 0

  def terminate(self):
    if not self.is_alive():
      return
    if hasattr(self, '_thread_id'):
      thread_id = self._thread_id
    else:
      thread_id = [k for k, v in threading._active.items() if v is self][0]
    result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread_id), ctypes.py_object(SystemExit))
    if result > 1:
      ctypes.pythonapi.PyThreadState_SetAsyncExc(
          ctypes.c_long(thread_id), None)
    print('Shut down worker:', self.name)


class Process:

  lock = None
  initializers = []

  def __init__(self, fn, *args, name=None):
    import multiprocessing
    import cloudpickle
    mp = multiprocessing.get_context('spawn')
    if Process.lock is None:
      Process.lock = mp.Lock()
    name = name or fn.__name__
    initializers = cloudpickle.dumps(self.initializers)
    args = (initializers,) + args
    self._process = mp.Process(
        target=self._wrapper, args=(Process.lock, fn, *args),
        name=name)

  def start(self):
    self._process.start()

  @property
  def name(self):
    return self._process.name

  @property
  def running(self):
    return self._process.is_alive()

  @property
  def pid(self):
    return self._process.pid

  @property
  def exitcode(self):
    return self._process.exitcode

  def terminate(self):
    self._process.terminate()
    print('Shut down worker:', self.name)

  def _wrapper(self, lock, fn, *args):
    try:
      import cloudpickle
      initializers, *args = args
      for initializer in cloudpickle.loads(initializers):
        initializer()
      fn(*args)
    except Exception:
      with lock:
        print('-' * 79)
        print(f'Exception in worker: {self.name}')
        print('-' * 79)
        print(''.join(traceback.format_exception(*sys.exc_info())))
      raise


def run(workers):
  for worker in workers:
    if not worker.running:
      worker.start()
  while True:
    if all(x.exitcode == 0 for x in workers):
      print('All workers terminated successfully.')
      return
    for worker in workers:
      if worker.exitcode not in (None, 0):
        # Wait for everybody who wants to print their error messages.
        time.sleep(1)
        [x.terminate() for x in workers if x is not worker]
        raise RuntimeError(f'Stopped workers due to crash in {worker.name}.')
    time.sleep(0.1)
