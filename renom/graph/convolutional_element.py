import renom as rm
from .core import operation, learnable_graph_element, operational_element, multi_gpu_variable
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

    inputs = inputs[0]['y']
    input_shape = inputs.shape
    
    self._inputs = inputs
    self._init = init.GlorotNormal()

    weight_shape = (self._channels, input_shape[1], self._kernel[0], self._kernel[1])
    bias_shape = (1, self._channels, 1, 1)
    
    self._conv_desc = rm.cuda.ConvolutionDescriptor(self._padding, self._stride, (1, 1), rm.precision)
    self._filter_desc = rm.cuda.FilterDescriptor(weight_shape, rm.precision)


    self._weights = multi_gpu_variable(shape = weight_shape, gpus = 1, initializer = self._init)
    self._bias = multi_gpu_variable(shape = bias_shape, gpus = 1)

    imgs = (input_shape[2] + self._padding[0] * 2 - self._kernel[0]) // self._stride[0] + 1
    output_shape = [input_shape[0], self._channels, imgs, imgs]
    self._outputs = multi_gpu_variable(shape = output_shape)
    self._vars = {'w' : self._weights, 'b' : self._bias, 'y' : self._outputs}

    with rm.cuda.RenomHandler() as handle:
      self._algo = rm.cuda.cuGetConvolutionFwdAlgo(handle, self._conv_desc, self._filter_desc, inputs[0], self._outputs[0])
      self._bwd_algo = rm.cuda.cuGetConvolutionBwdAlgo(handle, self._conv_desc, self._filter_desc, inputs[0], self._outputs[0])


  def perform(self):
    with rm.cuda.RenomHandler() as handle:
      rm.cuda.cuConvolutionForwardBiasActivation(handle, self._conv_desc, self._filter_desc, self._inputs[0], self._weights[0], self._outputs[0], self._bias[0], self._algo)


class convo_backward(operation):

  name = 'Convolution (B)'
  produces = ['w', 'b']

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):
    
    inputs = inputs[0]
    self._inputs = inputs
    self._fwd_w = self._fwd_op._weights
    self._fwd_b = self._fwd_op._bias
    self._fwd_in = self._fwd_op._inputs

    self._outputs = multi_gpu_variable(shape = self._fwd_in.shape)
    self._bias_out = multi_gpu_variable(shape = self._fwd_b.shape)
    self._weights_out = multi_gpu_variable(shape = self._fwd_w.shape)

    self._vars = {'w' : self._weights_out, 'b' : self._bias_out, 'y' : self._outputs}
    self._algo = self._fwd_op._bwd_algo

  def perform(self):
    with rm.cuda.RenomHandler() as handle:
      rm.cuda.cuActivationBackward(handle, self._fwd_op._outputs[0], self._inputs[0])
      rm.cuda.cuConvolutionBackward(handle, self._fwd_op._conv_desc, self._fwd_op._filter_desc, self._fwd_in[0], self._fwd_w[0], self._inputs[0], self._weights_out[0], self._bias_out[0], self._outputs[0], self._algo)

  def get_output_signature(self): return self._outputs




class ConvolutionalGraphElement(learnable_graph_element):
   
  has_back = True

  def __init__(self, channels = 3, kernel = 3, padding = 0, stride = 1, previous_element = None):

    self._chnls = channels
    self._krnl = kernel
    self._pdng = padding
    self._strd = stride
    fwd_op = convo_forward(channels, kernel, padding, stride)
    self._forward_operations = [ fwd_op ]
    self._backward_operations = [ convo_backward(fwd_op) ]

    super().__init__(previous_elements = previous_element)
