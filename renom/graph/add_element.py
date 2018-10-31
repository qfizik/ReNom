import renom as rm
import numpy as np
from operation import operation
from new_gpu import multi_gpu_variable
from graph_element import graph_element, operational_element
from learnable_graph import learnable_graph_element

class add(operation):
 
  name = 'Add (F)'
 
  def __init__(self):
    self._a = None
    self._b = None

  def setup(self, inputs):
    a = inputs[0]
    b = inputs[1]
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
  
  def get_output_signature(self): return self._c
  def __repr__(self): return self._c.__repr__()
  def as_ndarray(self): return self._c.as_ndarray()

class add_back(operation):

  name = 'Add (B)'

  def setup(self, inputs):
    self._outputs = inputs[0]

  def perform(self): pass

  def get_output_signature(self): return self._outputs


class add_element(learnable_graph_element):

  has_back = True

  def __init__(self):
  
    fwd_op = add()
    self._forward_operations = [ fwd_op ]
    self._backward_operations = [ add_back(), add_back() ]
    super().__init__()
 
def _add(self, other):
  ret = add_element()
  ret.connect(self, other)
  return ret

learnable_graph_element.__add__ = _add




