import renom as rm
from renom.graph.core import learnable_graph_element, operation, GraphFactory, graph_variable, multi_gpu_variable


class softplus_forward(operation):

  def setup(self, inputs, storage):
    inputs = inputs[0]['y']
    gpus = inputs.gpus
    self.gpus = gpus
    in_shape = inputs.shape
    outs = multi_gpu_variable(shape = in_shape, gpus = gpus)
    self._inputs = inputs
    self._outputs = outs
    self._vars = { 'y' : outs }

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cusoftplus_forward(self._inputs[gpu], self._outputs[gpu]) 

class softplus_backward(operation):

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):
    inputs = inputs[0]['y']
    gpus = inputs.gpus
    self.gpus = gpus
    in_shape = inputs.shape
    outs = multi_gpu_variable(shape = in_shape, gpus = gpus)
    self._inputs = inputs
    self._outputs = outs
    self._fwd_out = self._fwd_op._outputs
    self._vars = { 'y' : outs , id(self._fwd_op._inputs) : outs}

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cusoftplus_backward(self._fwd_out[gpu], self._inputs[gpu], self._outputs[gpu])


class SoftplusElement(learnable_graph_element):

  has_back = True

  def __init__(self, previous_elements = None):
    fwd_op = softplus_forward()
    bwd_ops = [ softplus_backward(fwd_op) ]
    super().__init__(forward_operation = fwd_op, backward_operations = bwd_ops, previous_elements = previous_elements)

class SoftplusGraphElement(GraphFactory):

  def connect(self, other):
    ret = SoftplusElement(previous_elements = other)
    return ret

