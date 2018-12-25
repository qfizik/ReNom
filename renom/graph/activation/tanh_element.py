import renom as rm
from renom.graph.core import learnable_graph_element, operation, GraphFactory, graph_variable, GraphMultiStorage
import renom.utility.initializer as init
import numpy as np

class tanh_forward(operation):

  def setup(self, inputs, storage):
    inputs = inputs[0]['y']
    gpus = inputs.gpus
    self.gpus = gpus
    outs = GraphMultiStorage(shape = inputs.shape, gpus = gpus)
    self._vars = { 'y' : outs}
    self._inputs = inputs
    self._outputs = outs

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cutanh(self._inputs[gpu], self._outputs[gpu])

class tanh_forward_cpu(tanh_forward):
  
  def perform(self):
    x = self._inputs['cpu']
    ret = np.tanh(x)
    self._outputs['cpu'] = ret

class tanh_backward(operation):

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):
    inputs = inputs[0]['y']
    gpus = inputs.gpus
    self.gpus = gpus
    outs = GraphMultiStorage(shape = inputs.shape, gpus = gpus)
    one = GraphMultiStorage(shape = (1, ), gpus = gpus, initializer = init.Constant(1))
    self._vars = { 'y' : outs, id(self._fwd_op._inputs) : outs}
    self._fwd_out = self._fwd_op._outputs
    self._inputs = inputs
    self._outputs = outs
    self._one = one

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cumul(self._fwd_out[gpu], self._fwd_out[gpu], self._outputs[gpu], handle)
      rm.cuda.cusub(self._one[gpu], self._outputs[gpu], self._outputs[gpu], handle)
      rm.cuda.cumul(self._inputs[gpu], self._outputs[gpu], self._outputs[gpu], handle)

class tanh_backward_cpu(tanh_backward):

  def perform(self):
    y = self._fwd_out['cpu']
    dy = self._inputs['cpu']
    ret = (1.0 - y ** 2) * dy
    self._outputs['cpu'] = ret


class TanhElement(learnable_graph_element):

  has_back = True

  def __init__(self, previous_elements = None):
    fwd_op = tanh_forward() if rm.is_cuda_active() else tanh_forward_cpu()
    bwd_ops = [ tanh_backward(fwd_op) if rm.is_cuda_active() else tanh_backward_cpu(fwd_op) ]
    super().__init__(forward_operation = fwd_op, backward_operations = bwd_ops, previous_elements = previous_elements)

class TanhGraphElement(GraphFactory):

  def __init__(self):
    super().__init__()

  def connect(self, other):
    ret = TanhElement(previous_elements = other)
    return ret

