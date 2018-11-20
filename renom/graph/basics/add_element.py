import renom as rm
from renom.graph.core import operation, multi_gpu_variable, operational_element, learnable_graph_element

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
    self._c = multi_gpu_variable(shape=a.shape, gpus=self.gpus)
    self._vars = { 'a' : a, 'b' : b, 'y' : self._c }

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cuadd(self._a[gpu], self._b[gpu], self._c[gpu], handle)
  

class add_back(operation):

  name = 'Add (B)'

  def setup(self, inputs, storage):
    self._outputs = inputs[0]['dy']
    self._vars = { 'y' : self._outputs, 'dy' : self._outputs } 

  def perform(self): pass



class AddElement(learnable_graph_element):

  _has_back = True
  _name = 'Add Element'
  

  def __init__(self, previous_elements = None):
  
    fwd_op = add_forward()
    bwd_ops = [ add_back(), add_back() ]
    super().__init__(fwd_op, bwd_ops, previous_elements)
 
def _add(self, other):
  ret = AddElement([self, other])
  return ret

learnable_graph_element.__add__ = _add




