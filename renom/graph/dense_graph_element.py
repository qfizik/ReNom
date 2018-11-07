from .core import learnable_graph_element, operational_element, operation, multi_gpu_variable
import renom.utility.initializer as init
import renom as rm

class dense_forward(operation):

  name = 'Dense (F)'
  consumes = ['w']

  def __init__(self, output_size):
    
    self._output_size = output_size

  def setup(self, inputs, storage):
    
    inputs = inputs[0]['y']
    assert isinstance(inputs, multi_gpu_variable), 'Received {}'.format(type(inputs))

    self.gpus = inputs.gpus
    self._init = init.GlorotNormal()

    self._inputs = inputs
    weight_shape = ( inputs[0].shape[1] , self._output_size )
    weights = multi_gpu_variable( shape = weight_shape , gpus = self.gpus, initializer = self._init)
    output_shape = ( inputs[0].shape[0] , self._output_size )
    outputs = multi_gpu_variable( shape = output_shape, gpus = self.gpus)

    self._vars = {'x' : inputs, 'w' : weights, 'y' : outputs}
    self._weights = weights
    self._outputs = outputs

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cublas_gemm(self._inputs[gpu], 0, self._weights[gpu], 0, self._outputs[gpu], handle)


class dense_backward(operation):

  name = 'Dense (B)'

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):

    inputs = inputs[0]['dy']
    gpus = inputs.gpus
    self.gpus = gpus
    weights = self._fwd_op.get_key('w')
    self._inputs = inputs
    self._weights = weights

    fwd_ins = self._fwd_op.get_key('x')
    output_shape = fwd_ins.shape

    outputs = multi_gpu_variable(shape = output_shape, gpus = gpus, initializer = None)

    self._vars = { 'y' : outputs, 'dy' : outputs }
    self._outputs = outputs

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cublas_gemm(self._inputs[gpu], 0, self._weights[gpu], 1, self._outputs[gpu], handle)



class dense_weight_backward(operation):

  name = 'Dense Weight (B)'
  produces = ['w']

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward


  def setup(self, inputs, storage):
    inputs = inputs[0]['dy']
    self._inputs = inputs

    gpus = inputs.gpus
    self.gpus = gpus
    fwd_ins = self._fwd_op.get_key('x')
    fwd_weights = self._fwd_op.get_key('w')
    output_shape = fwd_weights.shape

    outputs = multi_gpu_variable(shape = output_shape, gpus = gpus, initializer = None)

    self._vars = { 'y' : outputs, 'w' : outputs }

    self._fwd_ins = fwd_ins
    self._outputs = outputs
    
  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cublas_gemm(self._fwd_ins[gpu], 1, self._inputs[gpu], 0, self._outputs[gpu], handle)


class DenseGraphElement(learnable_graph_element):

  has_back = True

  def __init__(self, output_size, previous_element = None):
    
    fwd_op = dense_forward(output_size)
    self._forward_operations = [ fwd_op ]
    self._backward_operations = [ dense_backward(associated_forward = fwd_op),
                                  dense_weight_backward(associated_forward = fwd_op),]
    self._output_size = output_size

    super().__init__(previous_elements = previous_element)

  @property
  def weights(self):
    return self._fwd._op.get_key('w')

  @property
  def weights_back(self):
    return self._bwd_graphs[1].get_key('w')
