import renom as rm
from renom.graph.core import UserGraph, operation, GraphFactory, graph_variable, GraphMultiStorage
import numpy as np


class dropout_forward(operation):

  name = 'Dropout (F)'

  def __init__(self, dropout_rate = 0.5):
    self._dropout_rate = dropout_rate

  def setup(self, inputs):
    inputs = inputs[0]['y']
    gpus = inputs.gpus
    self.gpus = gpus
    mask = GraphMultiStorage(shape = inputs.shape, gpus = gpus)
    outs = GraphMultiStorage(shape = inputs.shape, gpus = gpus)
    self._vars = { 'y' : outs }
    self._inputs = inputs
    self._outputs = outs
    self._mask = mask

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.curand_generator().rand_bernoulli(self._mask[gpu], 1 - self._dropout_rate)
      rm.cuda.cudiv(self._mask[gpu], self._dropout_rate, self._mask[gpu], handle)
      rm.cuda.cumul(self._mask[gpu], self._inputs[gpu], self._outputs[gpu], handle)

class dropout_forward_cpu(dropout_forward):

  def perform(self):
    x = self._inputs['cpu']
    dropout_ratio = 1 - self._dropout_rate
    mask = np.array(np.random.rand(*x.shape) < dropout_ratio, dtype = rm.precision) / dropout_ratio 
    ret = x * mask
    self._mask['cpu'] = mask
    self._outputs['cpu'] = ret
    print(mask)
    print(ret)

class dropout_backward(operation):

  name = 'Dropout (B)'

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs):
    inputs = inputs[0]['y']
    gpus = inputs.gpus
    self.gpus = gpus
    outs = GraphMultiStorage(shape = inputs.shape, gpus = gpus)
    self._vars = {'y' : outs, id(self._fwd_op._inputs) : outs}
    self._fwd_mask = self._fwd_op._mask
    self._outputs = outs
    self._inputs = inputs

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cumul(self._inputs[gpu], self._fwd_mask[gpu], self._outputs[gpu], handle)
      
class dropout_backward_cpu(dropout_backward):

  def perform(self):

    dy = self._inputs['cpu']
    mask = self._fwd_mask['cpu']
    ret = dy * mask
    self._outputs['cpu'] = ret


class DropoutElement(UserGraph):

  has_back = True
  _inference = False

  def __init__(self, dropout_rate = 0.5, previous_elements = None):
    self.dropout_ratio = dropout_rate
    fwd_op = dropout_forward() if rm.is_cuda_active() else dropout_forward_cpu()
    bwd_ops = [ dropout_backward(fwd_op) if rm.is_cuda_active() else dropout_backward_cpu(fwd_op) ]
    super().__init__(forward_operation = fwd_op, backward_operations = bwd_ops, previous_elements = previous_elements)

  @property
  def inference(self):
    return self._inference

  @inference.setter
  def inference(self, val):
    self._inference = val

class DropoutGraphElement(GraphFactory):

  def connect(self, other):
    ret = DropoutElement(previous_elements = other)
    return ret
