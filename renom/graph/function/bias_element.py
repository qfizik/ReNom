from renom.graph.core import UserGraph, operation, GraphMultiStorage, GraphFactory, graph_variable
import renom.utility.initializer as init
import renom as rm
import numpy as np

class bias_forward(operation):

  name = 'Bias (F)'
  consumes = [ 'b' ]

  def __init__(self): pass

  def setup(self, inputs, storage):
    self._storage = storage

    bias = inputs[1]['y']
    inputs = inputs[0]['y']
    assert isinstance(inputs, GraphMultiStorage)
    self._inputs = inputs

    in_shape = inputs.shape
    bias_shape = ( 1 , in_shape[1] )
    gpus = inputs.gpus
    self.gpus = gpus
    self._init = init.Constant(0)
    bias.__init__( shape = bias_shape, gpus = self.gpus, initializer = self._init)
    outputs = GraphMultiStorage( shape = in_shape, gpus = self.gpus)
    self._vars = {'x' : inputs, 'b' : bias, 'y' : outputs} 
    self._outputs = outputs
    self._biases = bias

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cuadd(self._inputs[gpu], self._biases[gpu], self._outputs[gpu], handle)

class bias_forward_cpu(bias_forward):

  def perform(self):
    ret = self._inputs['cpu'] + self._biases['cpu']
    self._outputs['cpu'] = ret


class bias_backward(operation):

  name = 'Bias (B)'
  produces = [ 'b' ]

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):
    self._storage = storage
    inputs = inputs[0]['y']
    self.gpus = inputs.gpus
    self._bias_back = GraphMultiStorage(shape = self._fwd_op.get_key('b').shape, gpus = self.gpus)
    self._vars = { 'y' : inputs, 'dy' : inputs , 'b' : self._bias_back, id(self._fwd_op._biases) : self._bias_back}
    self._inputs = inputs

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      ret = rm.cuda.cusum(self._inputs[gpu], handle, axis = 0, keepdims = True)
      self._bias_back[gpu].copy_from(ret)

class bias_backward_cpu(bias_backward):

  def perform(self):
    ret = np.sum(self._inputs['cpu'], axis = 0, keepdims = True)
    self._bias_back['cpu'] = ret


class BiasElement(UserGraph):

  has_back = True

  def __init__(self, previous_element = None):
  
    fwd_op = bias_forward() if rm.is_cuda_active() else bias_forward_cpu()
    bwd_graphs = [ bias_backward(fwd_op) if rm.is_cuda_active() else bias_backward_cpu(fwd_op)]

    super().__init__(forward_operation = fwd_op, backward_operations = bwd_graphs, previous_elements = previous_element)


class BiasGraphElement(GraphFactory):

  def __init__(self):
    super().__init__()
    self.params['b'] = graph_variable()

  def connect(self, other):
    self.params['b'].disconnect()
    ret = BiasElement(previous_element = [ other, self.params['b']])
    return ret


