import renom as rm
from renom.graph.core import operation, GraphMultiStorage, operational_element, UserGraph

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
    self._c = GraphMultiStorage(shape=a.shape, gpus = self.gpus)
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
    self._outputs = GraphMultiStorage(shape = other.shape, gpus = other._num_gpus)
    self._vars = { 'y' : self._outputs, 'dy' : self._outputs }


  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cumul(self._dy[gpu], self._other[gpu], self._outputs[gpu], handle)



class MulElement(UserGraph):
  
  _has_back = True
  _name = 'Mul Element'

  def __init__(self, previous_elements = None):

    fwd_op = mul_forward()
    bwd_ops = [ mul_backward(fwd_op, 'b'), mul_backward(fwd_op, 'a') ]
    super().__init__(fwd_op, bwd_ops, previous_elements)

def _mul(self, other):
  ret = MulElement([self, other])
  return ret

UserGraph.__mul__ = _mul
