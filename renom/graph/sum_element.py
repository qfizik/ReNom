import renom as rm
from .core import learnable_graph_element, multi_gpu_variable, operation
import renom.utility.initializer as init

class sum_forward(operation):

  name = 'Sum (F)'

  def setup(self, inputs, storage):
    inputs = inputs[0]['y']
    self._inputs = inputs
    gpus = inputs.gpus
    self.gpus = gpus
    out_shape = ( 1, )
    outs = multi_gpu_variable(shape = out_shape, gpus = gpus)
    self._outputs = outs
    self._vars = { 'y' : outs }

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      r = rm.cuda.cusum(self._inputs[gpu], handle)
      self._outputs[gpu].copy_from(r)

class sum_backward(operation):

  name = 'Sum (B)'

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):
    inputs = inputs[0]['y']
    self._inputs = inputs
    gpus = inputs.gpus
    self.gpus = gpus
    out_shape = self._fwd_op._inputs.shape
    outs = multi_gpu_variable(shape = out_shape, gpus = gpus, initializer = init.Constant(1))
    self._outputs = outs
    self._vars = { 'y' : outs }

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      pass
      

class SumElement(learnable_graph_element):
  
  name = 'Sum'

  def __init__(self):
    self._forward_operations = [ sum_forward() ]
    self._backward_operations = [ ]
    super().__init__()



