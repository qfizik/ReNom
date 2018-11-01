from .core import learnable_graph_element, operational_element, operation, multi_gpu_variable
import renom.utility.initializer as init
import renom as rm

class bias_forward(operation):

  name = 'Bias (F)'
  consumes = [ 'b' ]

  def __init__(self): pass

  def setup(self, inputs, storage):
    self._storage = storage

    inputs = inputs[0]['y']
    assert isinstance(inputs, multi_gpu_variable)
    self._inputs = inputs

    in_shape = inputs.get_shape()
    bias_shape = ( 1 , in_shape[1] )
    gpus = inputs._num_gpus
    self._num_gpus = gpus
    self._init = init.Constant(0)
    bias = multi_gpu_variable( shape = bias_shape, gpus = self._num_gpus, initializer = self._init)
    outputs = multi_gpu_variable( shape = in_shape, gpus = self._num_gpus)
    self._vars = {'x' : inputs, 'b' : bias, 'y' : outputs} 
    self._outputs = outputs
    self._biases = bias

  def perform(self):
    for gpu, handle in enumerate(rm.cuda.RenomHandlers(self._num_gpus)):
      rm.cuda.cuadd(self._inputs[gpu], self._biases[gpu], self._outputs[gpu], handle)



class bias_backward(operation):

  name = 'Bias (B)'
  produces = [ 'b' ]

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):
    self._storage = storage
    self._num_gpus = self._fwd_op._num_gpus
    self._bias_back = multi_gpu_variable(shape = self._fwd_op.get_key('b').get_shape(), gpus = self._fwd_op._num_gpus)
    inputs = inputs[0]['dy']
    self._vars = { 'y' : inputs, 'dy' : inputs , 'b' : self._bias_back }
    self._inputs = inputs

  def perform(self):
    for gpu, handle in enumerate(rm.cuda.RenomHandlers(self._num_gpus)):
      ret = rm.cuda.cusum(self._inputs[gpu], handle, axis = 0, keepdims = True)
      self._bias_back[gpu] = ret


class BiasElement(learnable_graph_element):

  has_back = True

  def __init__(self, previous_element = None):
  
    fwd_op = bias_forward()
    self._forward_operations = [ fwd_op ]
    self._backward_operations = [ bias_backward(fwd_op) ]

    super().__init__(previous_elements = previous_element)




