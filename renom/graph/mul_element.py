import renom as rm
from .core import operation, multi_gpu_variable, operational_element, learnable_graph_element

class mul_forward(operation):
  
  name = 'Mul (F)'

  def __init__(self): pass

  def setup(self, inputs, storage):
    a = inputs[0]['y']
    b = inputs[1]['y']
    assert len(a) == len(b)
    for _a, _b in zip(a, b):
      assert _a.shape == _b.shape 
    self.gpus = a.gpus
    self._a = a
    self._b = b
    self._c = multi_gpu_variable(shape=a.shape, gpus = self.gpus)
    self._vars = { 'a' : a, 'b' : b, 'y' : self._c }

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cumul(self._a[gpu], self._b[gpu], self._c[gpu], handle)


class mul_backward(operation):

  name = 'Mul (B)'

  def __init__(self, associated_forward, key):
    self._fwd_op = associated_forward
    self._key = key

  def setup(self, inputs, storage):
    self._dy = inputs[0]['dy']
    other = self._fwd_op.get_key(self._key)
    self._other = other
    self._outputs = multi_gpu_variable(shape = other.shape, gpus = other._num_gpus)
    self._vars = { 'y' : self._outputs, 'dy' : self._outputs }


  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cumul(self._dy[gpu], self._other[gpu], self._outputs[gpu], handle)



class MulElement(learnable_graph_element):
  
  _has_back = True
  _name = 'Mul Element'

  def __init__(self):

    fwd_op = mul_forward()
    self._forward_operations = [ fwd_op ]
    self._backward_operations = [ mul_backward(fwd_op, 'b'), mul_backward(fwd_op, 'a') ]
    super().__init__()

def _mul(self, other):
  ret = MulElement()
  ret.connect(self, other)
  return ret

learnable_graph_element.__mul__ = _mul
