import numpy as np
import renom as rm
import contextlib as cl
import collections

_renom_handlers = {}

@cl.contextmanager
def RenomHandler(device = None):
  if device is None:
    device = rm.cuda.cuGetDevice()
  with rm.cuda.use_device(device):
    if device not in _renom_handlers:
      _renom_handlers[device] = RenomHandle(device)
    yield _renom_handlers[device]

class RenomHandle:

  def __init__(self, device=None, prefetch_length = 4):
    assert rm.cuda.is_cuda_active(), 'Cuda should be active before building cuda-related objects.'
    self.device = rm.cuda.cuGetDevice()
    with rm.cuda.use_device(self.device):
      self.stream = rm.cuda.cuCreateStream()
      self.memstream = rm.cuda.cuCreateStream()
    self.pinned_memory = {}
    self.cublas_handler = rm.cuda.createCublasHandle(self.stream)
    self.cudnn_handler = rm.cuda.createCudnnHandle(self.stream)
    self.prefetch_length = prefetch_length

  def getPinnedMemory(self, array):
    if array.shape not in self.pinned_memory:
      self._preparePins(array)
    ret = self.pinned_memory[array.shape][0]
    self.pinned_memory[array.shape].rotate(-1)
    ret.pin(array)
    return ret

  def _preparePins(self, array):
    self.pinned_memory[array.shape] = collections.deque(maxlen = self.prefetch_length)
    for pin in range(self.prefetch_length):
      self.pinned_memory[array.shape].append(rm.cuda.PinnedMemory(array, self.memstream))

def RenomHandlers(gpus):
  if isinstance(gpus, int):
    gpus = range(gpus)
  for gpu in gpus:
    with RenomHandler(gpu) as handle:
      yield gpu, handle
