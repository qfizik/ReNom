import renom as rm
from renom.graph.core import learnable_graph_element, operation, GraphFactory, graph_variable, multi_gpu_variable


class selu_forward(operation):

  def __init__(self):
    alpha = 1.6732632423543772848170429916717
    lamda = 1.0507009873554804934193349852946
    self._alpha = alpha
    self._lamda = lamda

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
      rm.cuda.cueru_forward(self._alpha, self._inputs[gpu], self._outputs[gpu]) 
      rm.cuda.cumul(self._outputs[gpu], self._lamda, self._outputs[gpu], handle)

class selu_backward(operation):

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward
    self._alpha = self._fwd_op._alpha
    self._lamda = self._fwd_op._lamda

  def setup(self, inputs, storage):
    inputs = inputs[0]['y']
    gpus = inputs.gpus
    self.gpus = gpus
    in_shape = inputs.shape
    outs = multi_gpu_variable(shape = in_shape, gpus = gpus)
    self._inputs = inputs
    self._outputs = outs
    self._fwd_in = self._fwd_op._inputs
    self._vars = { 'y' : outs , id(self._fwd_op._inputs) : outs}

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cueru_backward(self._alpha, self._fwd_in[gpu], self._outputs[gpu])
      rm.cu.cumul(self._outputs[gpu], self._inputs[gpu], self._outputs[gpu], handle)
      rm.cuda.cumul(self._outputs[gpu], self._lamda, self._outputs[gpu], handle)


class SeluElement(learnable_graph_element):

  has_back = True

  def __init__(self, previous_elements = None):
    fwd_op = selu_forward()
    bwd_ops = [ selu_backward(fwd_op) ]
    super().__init__(forward_operation = fwd_op, backward_operations = bwd_ops, previous_elements = previous_elements)

class SeluGraphElement(GraphFactory):

  def __init__(self, ):
    super().__init__()

  def connect(self, other):
    ret = SeluElement(previous_elements = other)
    return ret

