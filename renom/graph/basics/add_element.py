import renom as rm
from renom.graph.core import operation, GraphMultiStorage, operational_element, learnable_graph_element
import numpy as np

class add_forward(operation):
 
  name = 'Add (F)'
 
  def __init__(self):
    self._a = None
    self._b = None

  def setup(self, inputs, storage):
    a = inputs[0]['y']
    b = inputs[1]['y']
    assert len(a) == len(b)
    for _a, _b in zip(a, b):
      assert _a.shape == _b.shape
    self.gpus = a.gpus
    self._a = a
    self._b = b
    self._c = GraphMultiStorage(shape=a.shape, gpus=self.gpus)
    self._vars = { 'a' : a, 'b' : b, 'y' : self._c }

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cuadd(self._a[gpu], self._b[gpu], self._c[gpu], handle)
  
class add_forward_cpu(add_forward):
  
  def perform(self):
    a = self._a['cpu']
    b = self._b['cpu']
    self._c['cpu'] = a + b

class add_back(operation):

  name = 'Add (B)'

  def __init__(self, associated_forward, key):
    self._fwd_op = associated_forward
    self._key = key

  def setup(self, inputs, storage):
    self._outputs = inputs[0]['dy']
    self._vars = { 'y' : self._outputs, 'dy' : self._outputs, id(self._fwd_op.get_key(self._key)) : self._outputs } 

  def perform(self): pass



class AddElement(learnable_graph_element):

  _has_back = True
  _name = 'Add Element'
  

  def __init__(self, previous_elements = None):
  
    fwd_op = add_forward() if rm.is_cuda_active() else add_forward_cpu()
    bwd_ops = [ add_back(fwd_op, 'a'), add_back(fwd_op, 'b') ]
    super().__init__(fwd_op, bwd_ops, previous_elements)
 
def _add(self, other):
  ret = AddElement([self, other])
  return ret

learnable_graph_element.__add__ = _add




