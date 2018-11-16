import renom as rm
from renom.graph.core import operation, learnable_graph_element, multi_gpu_variable, GraphFactory, graph_variable
import renom.utility.initializer as init

class convo_forward(operation):

  name = 'Convolution (F)'
  consumes = ['w', 'b']

  def __init__(self, channels, kernel = 3, padding = 0, stride = 1):
    self._channels = channels
    self._kernel = (kernel, kernel)
    self._padding = (padding, padding)
    self._stride = (stride, stride)


  def setup(self, inputs, storage):

    weights = inputs[1]['y']
    bias = inputs[2]['y']
    inputs = inputs[0]['y']
    input_shape = inputs.shape
    
    self._inputs = inputs
    self._init = init.GlorotNormal()
    gpus = inputs.gpus
    self.gpus = gpus

    weight_shape = (self._channels, input_shape[1], self._kernel[0], self._kernel[1])
    bias_shape = (1, self._channels, 1, 1)
    
    self._conv_desc = rm.cuda.ConvolutionDescriptor(self._padding, self._stride, (1, 1), rm.precision)
    self._filter_desc = rm.cuda.FilterDescriptor(weight_shape, rm.precision)
    
    if weights.ready is False:
      weights.__init__(shape = weight_shape, gpus = gpus, initializer = self._init)
    else:
      assert weights.shape == weight_shape
    if bias.ready is False:
      bias.__init__(shape = bias_shape, gpus = gpus, initializer = init.Constant(0))
    else:
      assert bias.shape == bias_shape

    self._weights = weights
    self._bias = bias

    imgs = (input_shape[2] + self._padding[0] * 2 - self._kernel[0]) // self._stride[0] + 1
    output_shape = [input_shape[0], self._channels, imgs, imgs]
    self._outputs = multi_gpu_variable(shape = output_shape, gpus = gpus)
    self._vars = {'w' : self._weights, 'b' : self._bias, 'y' : self._outputs}

    with rm.cuda.RenomHandler() as handle:
      self._algo = rm.cuda.cuGetConvolutionFwdAlgo(handle, self._conv_desc, self._filter_desc, inputs[0], self._outputs[0])
      self._bwd_algo = rm.cuda.cuGetConvolutionBwdAlgo(handle, self._conv_desc, self._filter_desc, inputs[0], self._outputs[0])


  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cuConvolutionForwardBiasActivation(handle, self._conv_desc, self._filter_desc, self._inputs[gpu], self._weights[gpu], self._outputs[gpu], self._bias[gpu], self._algo)


class convo_backward(operation):

  name = 'Convolution (B)'
  produces = ['w', 'b']

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):
    
    inputs = inputs[0]['y']
    self._inputs = inputs
    self._fwd_w = self._fwd_op._weights
    self._fwd_b = self._fwd_op._bias
    self._fwd_in = self._fwd_op._inputs
    self.gpus = inputs.gpus

    self._outputs = multi_gpu_variable(shape = self._fwd_in.shape, gpus = self.gpus)
    self._bias_out = multi_gpu_variable(shape = self._fwd_b.shape, gpus = self.gpus)
    self._weights_out = multi_gpu_variable(shape = self._fwd_w.shape, gpus = self.gpus)

    self._vars = { 'w' : self._weights_out, 'b' : self._bias_out, 'y' : self._outputs,
                  id(self._fwd_in) : self._outputs,
                  id(self._fwd_w) : self._weights_out,
                  id(self._fwd_b) : self._bias_out,
                 }
    self._algo = self._fwd_op._bwd_algo

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cuActivationBackward(handle, self._fwd_op._outputs[gpu], self._inputs[gpu])
      rm.cuda.cuConvolutionBackward(handle, self._fwd_op._conv_desc, self._fwd_op._filter_desc, self._fwd_in[gpu], self._fwd_w[gpu], self._inputs[gpu], self._weights_out[gpu], self._bias_out[gpu], self._outputs[gpu], self._algo)





class ConvolutionalGraph(learnable_graph_element):
   
  has_back = True

  def __init__(self, channels = 3, kernel = 3, padding = 0, stride = 1, previous_element = None):

    self._chnls = channels
    self._krnl = kernel
    self._pdng = padding
    self._strd = stride
    fwd_op = convo_forward(channels, kernel, padding, stride)
    bwd_ops = [ convo_backward(fwd_op) ]

    super().__init__(forward_operation = fwd_op, backward_operations = bwd_ops, previous_elements = previous_element)


class ConvolutionalGraphElement(GraphFactory):

  def __init__(self, channels = 3, kernel = 3, padding = 0, stride = 1):
    self._chnls = channels
    self._krnl = kernel
    self._pdng = padding
    self._strd = stride
    self._weights = graph_variable()
    self._bias = graph_variable()

  @property
  def weights(self): return self._weights.output

  @property
  def bias(self): return self._bias.output

  def connect(self, other):
    ret = ConvolutionalGraph(self._chnls, self._krnl, self._pdng, self._strd, previous_element = [ other, self._weights, self._bias ])
    return ret
