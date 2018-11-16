import renom as rm
from renom.graph.core import learnable_graph_element, operation, GraphFactory, multi_gpu_variable, graph_variable
import renom.utility.initializer as init

class batch_norm_forward(operation):

  name = 'Batch Normalize (F)'

  def __init__(self, momentum = 0.99, mode = 0, epsilon = 1e-5):
    self._momentum = momentum
    self._mode = mode
    self._epsilon = epsilon
    self._inference = False 


  def setup(self, inputs, storage):
    bias = inputs[2]['y']
    weights = inputs[1]['y']
    inputs = inputs[0]['y']
    gpus = inputs.gpus
    self.gpus = gpus
   
    in_shape = inputs.shape
    weight_shape = tuple([1,] + list(in_shape[1:]))
    bias_shape = weight_shape

    weights.__init__(shape = weight_shape, gpus = gpus, initializer = init.GlorotNormal())
    bias.__init__(shape = bias_shape, gpus = gpus, initializer = init.Constant(0))
    outs = multi_gpu_variable(shape = in_shape, gpus = gpus)
    mean = multi_gpu_variable(shape = weight_shape, gpus = gpus)
    sq_var = multi_gpu_variable(shape = weight_shape, gpus = gpus)
    mv_m = multi_gpu_variable(shape = weight_shape, gpus = gpus, initializer = init.Constant(0))
    mv_v = multi_gpu_variable(shape = weight_shape, gpus = gpus, initializer = init.Constant(0))

    self._inputs = inputs
    self._weights = weights
    self._bias = bias
    self._outputs = outs
    self._mean = mean
    self._sq_var = sq_var
    self._mv_m = mv_m
    self._mv_v = mv_v
    self._vars = { 'y' : outs, 'w' : weights, 'b' : bias } 



  def perform(self): 
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      #rm.cuda.cusub(self._mv_m[gpu], self._mv_m[gpu], self._mv_m[gpu], handle)
      #rm.cuda.cusub(self._mv_v[gpu], self._mv_v[gpu], self._mv_v[gpu], handle)
      rm.cuda.cuBatchNormalizatoinForward(handle, self._inputs[gpu], self._mv_m[gpu], self._mv_v[gpu], self._weights[gpu], self._bias[gpu], self._outputs[gpu], self._mean[gpu], self._sq_var[gpu], self._momentum, self._mode, self._inference, self._epsilon)

class batch_norm_backward(operation):

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):
    inputs = inputs[0]['y']
    gpus = inputs.gpus
    self.gpus = gpus

    self._inputs = inputs
    self._fwd_ins = self._fwd_op._inputs
    self._fwd_w = self._fwd_op._weights
    self._mean = self._fwd_op._mean
    self._var = self._fwd_op._sq_var
    self._outputs = multi_gpu_variable(shape = inputs.shape, gpus = self.gpus, initializer = init.Constant(0))
    self._weights_back = multi_gpu_variable(shape = self._fwd_w.shape, gpus = self.gpus, initializer = init.Constant(1))
    self._bias_back = multi_gpu_variable(shape = self._fwd_op._bias.shape, gpus = self.gpus, initializer = init.Constant(1))
    self._vars = { 'y' : self._outputs, 'dy' : self._outputs, 'w' : self._weights_back, 'b' : self._bias_back, id(self._fwd_w) : self._weights_back, id(self._fwd_ins) : self._outputs, id(self._fwd_op._bias) : self._bias_back }
  
  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cuBatchNormalizatoinBackward(handle, self._fwd_ins[gpu], self._fwd_w[gpu], self._inputs[gpu], self._mean[gpu], self._var[gpu], self._outputs[gpu], self._weights_back[gpu], self._bias_back[gpu], 0)

    

class BatchNormalizer(learnable_graph_element):

  has_back = True

  def __init__(self, previous_elements = None):
    fwd_op = batch_norm_forward()
    bwd_ops = [ batch_norm_backward(fwd_op) ]
    super().__init__(forward_operation = fwd_op, backward_operations = bwd_ops, previous_elements = previous_elements)


class BatchNormalizeElement(GraphFactory):

  def __init__(self):
    self._weights = graph_variable()
    self._bias = graph_variable()

  @property
  def weights(self):
    return self._weights.output

  def connect(self, other):
    ret = BatchNormalizer(previous_elements = [ other, self._weights, self._bias ])
    return ret
