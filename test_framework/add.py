import renom as rm
import numpy as np
from operation import operation
from new_gpu import multi_gpu_variable

class add(operation):
  
  def __init__(self):
    self._a = None
    self._b = None
    self._c = None

  def setup(self, a, b):
    assert len(a) == len(b)
    for _a, _b in zip(a, b):
      assert _a.shape == _b.shape
    self._num_gpus = len(a)
    self._gpus = [gpu for gpu in range(self._num_gpus)]
    self._a = a
    self._b = b
    self._c = multi_gpu_variable(shape=a[0].shape, gpus=self._num_gpus, allocate_backward=True) 

  def perform(self):
    for gpu, handle in enumerate(rm.cuda.RenomHandlers(self._gpus)):
      rm.cuda.cuadd(self._a[gpu], self._b[gpu], self._c[gpu], handle)
  
  def get_output(self): return self._c


class add_back(operation):

  def setup(self, a):
    self._a = a

  def perform(self): pass

  def get_output(self): return self._a


