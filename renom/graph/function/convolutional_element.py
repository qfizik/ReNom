import renom as rm
from renom.layers.function.utils import im2col, col2im
from renom.graph.core import operation, learnable_graph_element, multi_gpu_variable, GraphFactory, graph_variable
import renom.utility.initializer as init
import numpy as np

class convo_forward(operation):

  name = 'Convolution (F)'
  consumes = ['w', 'b']

  def __init__(self, channels, kernel = 3, padding = 0, stride = 1):
    self._channels = channels
    self._kernel = (kernel, kernel)
    self._padding = (padding, padding)
    self._stride = (stride, stride)
    self._dilation = (1, 1)


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
    
    
    weights.__init__(shape = weight_shape, gpus = gpus, initializer = self._init)
    bias.__init__(shape = bias_shape, gpus = gpus, initializer = init.Constant(0))

    self._weights = weights
    self._bias = bias

    imgs = (input_shape[2] + self._padding[0] * 2 - self._kernel[0]) // self._stride[0] + 1
    output_shape = [input_shape[0], self._channels, imgs, imgs]
    self._outputs = multi_gpu_variable(shape = output_shape, gpus = gpus)
    self._vars = {'w' : self._weights, 'b' : self._bias, 'y' : self._outputs}

    if rm.is_cuda_active():
      with rm.cuda.RenomHandler() as handle:
        self._conv_desc = rm.cuda.ConvolutionDescriptor(self._padding, self._stride, (1, 1), rm.precision)
        self._filter_desc = rm.cuda.FilterDescriptor(weight_shape, rm.precision)
        self._algo = rm.cuda.cuGetConvolutionFwdAlgo(handle, self._conv_desc, self._filter_desc, inputs[0], self._outputs[0])
        self._bwd_algo = rm.cuda.cuGetConvolutionBwdAlgo(handle, self._conv_desc, self._filter_desc, inputs[0], self._outputs[0])


  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cuConvolutionForwardBiasActivation(handle, self._conv_desc, self._filter_desc, self._inputs[gpu], self._weights[gpu], self._outputs[gpu], self._bias[gpu], self._algo)

class convo_forward_cpu(convo_forward):

  def perform(self):
    col = im2col(self._inputs['cpu'], self._outputs.shape[2:], self._kernel, self._stride, self._padding, self._dilation)
    self._col = col
    val = np.rollaxis(np.tensordot(col, self._weights['cpu'], ([1, 2, 3], [1, 2, 3])), 3, 1)
    val = val + self._bias['cpu']
    self._outputs['cpu'] = val


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

    if rm.is_cuda_active():
      self._algo = self._fwd_op._bwd_algo

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cuActivationBackward(handle, self._fwd_op._outputs[gpu], self._inputs[gpu])
      rm.cuda.cuConvolutionBackward(handle, self._fwd_op._conv_desc, self._fwd_op._filter_desc, self._fwd_in[gpu], self._fwd_w[gpu], self._inputs[gpu], self._weights_out[gpu], self._bias_out[gpu], self._outputs[gpu], self._algo)


class convo_backward_cpu(convo_backward):

  def perform(self):
    dy = self._inputs['cpu']

    dx = np.tensordot(self._fwd_w['cpu'], dy, (0, 1))
    dx = np.rollaxis(dx, 3)
    dx = col2im(dx, self._fwd_in.shape[2:], self._fwd_op._stride, self._fwd_op._padding, self._fwd_op._dilation)
    self._outputs['cpu'] = dx

    dw = np.tensordot(dy, self._fwd_op._col, ([0, 2, 3], [0, 4, 5]))
    self._weights_out['cpu'] = dw
    
    db = np.sum(dy, (0, 2, 3), keepdims = True)
    self._bias_out['cpu'] = db



class ConvolutionalGraph(learnable_graph_element):
   
  has_back = True

  def __init__(self, channels = 3, kernel = 3, padding = 0, stride = 1, previous_element = None):

    self._chnls = channels
    self._krnl = kernel
    self._pdng = padding
    self._strd = stride
    fwd_op = convo_forward(channels, kernel, padding, stride) if rm.is_cuda_active() else convo_forward_cpu(channels, kernel, padding, stride)
    bwd_ops = [ convo_backward(fwd_op) if rm.is_cuda_active() else convo_backward_cpu(fwd_op) ]

    super().__init__(forward_operation = fwd_op, backward_operations = bwd_ops, previous_elements = previous_element)


class ConvolutionalGraphElement(GraphFactory):

  def __init__(self, channels = 3, kernel = 3, padding = 0, stride = 1):
    super().__init__()
    self._chnls = channels
    self._krnl = kernel
    self._pdng = padding
    self._strd = stride
    self.params['w'] = graph_variable()
    self.params['b'] = graph_variable()


  def connect(self, other):
    ret = ConvolutionalGraph(self._chnls, self._krnl, self._pdng, self._strd, previous_element = [ other, self.params['w'], self.params['b']])
    return ret
