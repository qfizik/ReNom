from renom.graph.core import learnable_graph_element, operation, multi_gpu_variable, GraphFactory, graph_variable
import renom.utility.initializer as init
import renom as rm

class bias_forward(operation):

  name = 'Bias (F)'
  consumes = [ 'b' ]

  def __init__(self): pass

  def setup(self, inputs, storage):
    self._storage = storage

    bias = inputs[1]['y']
    inputs = inputs[0]['y']
    assert isinstance(inputs, multi_gpu_variable)
    self._inputs = inputs

    in_shape = inputs.shape
    bias_shape = ( 1 , in_shape[1] )
    gpus = inputs.gpus
    self.gpus = gpus
    self._init = init.Constant(0)
    bias.__init__( shape = bias_shape, gpus = self.gpus, initializer = self._init)
    outputs = multi_gpu_variable( shape = in_shape, gpus = self.gpus)
    self._vars = {'x' : inputs, 'b' : bias, 'y' : outputs} 
    self._outputs = outputs
    self._biases = bias

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cuadd(self._inputs[gpu], self._biases[gpu], self._outputs[gpu], handle)



class bias_backward(operation):

  name = 'Bias (B)'
  produces = [ 'b' ]

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):
    self._storage = storage
    inputs = inputs[0]['y']
    self.gpus = inputs.gpus
    self._bias_back = multi_gpu_variable(shape = self._fwd_op.get_key('b').shape, gpus = self.gpus)
    self._vars = { 'y' : inputs, 'dy' : inputs , 'b' : self._bias_back, id(self._fwd_op._biases) : self._bias_back}
    self._inputs = inputs

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      ret = rm.cuda.cusum(self._inputs[gpu], handle, axis = 0, keepdims = True)
      self._bias_back[gpu].copy_from(ret)


class BiasElement(learnable_graph_element):

  has_back = True

  def __init__(self, previous_element = None):
  
    fwd_op = bias_forward()
    bwd_graphs = [ bias_backward(fwd_op) ]

    super().__init__(forward_operation = fwd_op, backward_operations = bwd_graphs, previous_elements = previous_element)


class BiasGraphElement(GraphFactory):

  def __init__(self):
    super().__init__()
    self.params['b'] = graph_variable()

  def connect(self, other):
    ret = BiasElement(previous_element = [ other, self.params['b']])
    return ret


