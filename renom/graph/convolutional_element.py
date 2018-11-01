import renom as rm
from operation import operation
from learnable_graph import learnable_graph_element
from graph_element import operational_element
from new_gpu import multi_gpu_variable
import renom.utility.initializer as init
from update_graph import update_operation

class convo_forward(operation):

  name = 'Convolution (F)'
  consumes = ['w', 'b']

  def __init__(self, channels, kernel = 3, padding = 0, stride = 1):
    self._channels = channels
    self._kernel = (kernel, kernel)
    self._padding = (padding, padding)
    self._stride = (stride, stride)
    self._setup = False


  def setup(self, inputs):
    self._setup = True

    inputs = inputs[0]
    input_shape = inputs.get_shape()
    
    self._inputs = inputs
    self._init = init.GlorotNormal()

    weight_shape = (self._channels, input_shape[1], self._kernel[0], self._kernel[1])
    bias_shape = (1, self._channels, 1, 1)
    
    self._conv_desc = rm.cuda.ConvolutionDescriptor(self._padding, self._stride, (1, 1), rm.precision)
    self._filter_desc = rm.cuda.FilterDescriptor(weight_shape, rm.precision)

    self._algos = {'Forward' : 0, 'Backward' : {'Data' : 0, 'Filter' : 0}}

    self._weights = multi_gpu_variable(shape = weight_shape, gpus = 1, initializer = self._init)
    self._bias = multi_gpu_variable(shape = bias_shape, gpus = 1)

    self._vars = {'w' : self._weights, 'b' : self._bias}
    imgs = (input_shape[2] + self._padding[0] * 2 - self._kernel[0]) // self._stride[0] + 1
    output_shape = [input_shape[0], self._channels, imgs, imgs]
    self._outputs = multi_gpu_variable(shape = output_shape)


  def perform(self):
    with rm.cuda.RenomHandler() as handle:
      rm.cuda.cuConvolutionForwardBiasActivation(handle, self._conv_desc, self._filter_desc, self._inputs[0], self._weights[0], self._outputs[0], self._bias[0], 0)

  def get_output_signature(self): return self._outputs

  def __repr__(self): return self._outputs.__repr__()

class convo_backward(operation):

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs):
    
    inputs = inputs[0]
    self._inputs = inputs
    self._fwd_w = self._fwd_op._weights
    self._fwd_b = self._fwd_op._bias
    self._fwd_in = self._fwd_op._inputs

    self._outputs = multi_gpu_variable(shape = self._fwd_in.get_shape())
    self._bias_out = multi_gpu_variable(shape = self._fwd_b.get_shape())
    self._weights_out = multi_gpu_variable(shape = self._fwd_w.get_shape())

    self._vars = {'w' : self._weights_out, 'b' : self._bias_out}


  def perform(self):
    with rm.cuda.RenomHandler() as handle:
      rm.cuda.cuActivationBackward(handle, self._fwd_op._outputs[0], self._inputs[0])
      rm.cuda.cuConvolutionBackward(handle, self._fwd_op._conv_desc, self._fwd_op._filter_desc, self._fwd_in[0], self._fwd_w[0], self._inputs[0], self._weights_out[0], self._bias_out[0], self._outputs[0], {'data' : 0, 'filter' : 0})

  def get_output_signature(self): return self._outputs

  def __repr__(self): return self._outputs.__repr__()



class ConvolutionalGraphElement(learnable_graph_element):
   
  has_back = True

  def __init__(self, channels = 3, kernel = 3, padding = 0, stride = 1, previous_element = None):

    self._chnls = channels
    self._krnl = kernel
    self._pdng = padding
    self._strd = stride

    super().__init__(previous_elements = previous_element)

  def connect(self, previous_element):

    forward_operation = convo_forward(self._chnls, self._krnl, self._pdng, self._strd)
    forward_graph = operational_element(forward_operation, tags = ['Forward'])

    prev_graph_input = previous_element.get_forward_output()
    forward_graph.add_input(prev_graph_input)

    backward_operation = convo_backward(forward_operation)
    backward_graph = operational_element(backward_operation, tags = ['Backward'])

    weight_update = update_operationn(0.01, forward_operation, backward_operation, 'w')
    update_graphA = operational_element(weight_update, tags = ['Update'])
    update_graphA.add_input(backward_graph)   
    bias_update = update_operationn(0.01, forward_operation, backward_operation, 'b') 
    update_graphB = operational_element(bias_update, tags = ['Update'])
    update_graphB.add_input(backward_graph)

    self._fwd = forward_graph
    self._bwd = backward_graph

    if previous_element.has_back:
      previous_element.connect_back(self)

  def connect_back(self, previous_element):
    backward_graph_input = previous_element.get_backward_output()

    self._bwd.add_input(backward_graph_input)

  def __repr__(self): return self._fwd.__repr__()


  def forward(self):
    self._fwd.forward(tag = 'Forward')
  
  def get_forward_output(self): return self._fwd
  def get_backward_output(self): return self._bwd


