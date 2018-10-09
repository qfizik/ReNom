import numpy as np
import renom as rm
import contextlib as cl

_renom_handlers = {}

@cl.contextmanager
def RenomHandler(device = 0):
  with rm.cuda.use_device(device):
    if device not in _renom_handlers:
      _renom_handlers[device] = RenomHandle(device)
    yield _renom_handlers[device]

class RenomHandle:

  def __init__(self, device=None):
    assert rm.cuda.is_cuda_active(), 'Cuda should be active before building cuda-related objects.'
    self.device = rm.cuda.cuGetDevice()
    with rm.cuda.use_device(self.device):
      self.stream = rm.cuda.cuCreateStream()
    self.pinned_memory = {}
    self.cublas_handler = rm.cuda.createCublasHandle(self.stream)
    self.cudnn_handler = None

  def pin(self, array):
    if isinstance(array, np.ndarray):
        
