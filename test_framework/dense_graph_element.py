from learnable_graph import learnable_graph_element
from graph_element import operational_element
from operation import operation
from update_graph import update_operation
from new_gpu import multi_gpu_variable
import renom.utility.initializer as init
import renom as rm

class dense_forward(operation):

  name = 'Dense (F)'

  def __init__(self, output_size):
    
    self._output_size = output_size
    self._outputs = multi_gpu_variable()

  def setup(self, inputs):
    
    inputs = inputs[0]
    assert isinstance(inputs, multi_gpu_variable)

    self._init = init.GlorotNormal()

    self._inputs = inputs
    weight_shape = ( inputs[0].shape[1] , self._output_size )
    weights = multi_gpu_variable( shape = weight_shape , gpus = 1, initializer = None)
    output_shape = ( inputs[0].shape[0] , self._output_size )

    self._weights = weights
    self._outputs.__init__(shape = output_shape, gpus = 1)

  def perform(self):
    with rm.cuda.RenomHandler() as handle:
      rm.cuda.cublas_gemm(self._inputs[0], 0, self._weights[0], 0, self._outputs[0], handle)

  def get_output_signature(self):
    assert self._outputs is not None
    return self._outputs

  def get_weights_signature(self): return self._weights
  def get_input_signature(self): return self._inputs

  def __repr__(self):
    return self._outputs.__repr__()


class dense_backward(operation):

  name = 'Dense (B)'

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs):

    inputs = inputs[0]
    weights = self._fwd_op.get_weights_signature()
    self._inputs = inputs
    self._weights = weights

    fwd_ins = self._fwd_op.get_input_signature()
    output_shape = fwd_ins.get_shape()

    outputs = multi_gpu_variable(shape = output_shape, gpus = 1, initializer = None)

    self._outputs = outputs

  def perform(self):
    with rm.cuda.RenomHandler() as handle:
      rm.cuda.cublas_gemm(self._inputs[0], 0, self._weights[0], 1, self._outputs[0], handle)

  def get_output_signature(self): return self._outputs


class dense_weight_backward(operation):

  name = 'Dense Weight (B)'

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward


  def setup(self, inputs):
    inputs = inputs[0]
    self._inputs = inputs

    fwd_ins = self._fwd_op.get_input_signature()
    fwd_weights = self._fwd_op.get_weights_signature()
    output_shape = fwd_weights.get_shape()

    outputs = multi_gpu_variable(shape = output_shape, gpus = 1, initializer = None)

    self._fwd_ins = fwd_ins
    self._outputs = outputs
    
  def perform(self):
    with rm.cuda.RenomHandler() as handle:
      rm.cuda.cublas_gemm(self._fwd_ins[0], 1, self._inputs[0], 0, self._outputs[0], handle)

  def get_output_signature(self): return self._outputs

class dense_graph_element(learnable_graph_element):

  has_back = True

  def __init__(self, output_size, previous_element = None):
     
    self._output_size = output_size
    if previous_element is not None:
      self.connect(previous_element)

    super().__init__(previous_elements = previous_element)

  def connect(self, previous_element):

    forward_operation = dense_forward(self._output_size)
    forward_graph =  operational_element(operation = forward_operation, tags = ['Forward'])

    prev_graph_input = previous_element.get_forward_output()
    forward_graph.add_input(previous_element.get_forward_output())

    backward_operation = dense_backward(associated_forward = forward_operation)
    backward_graph = operational_element(backward_operation, tags = ['Backward'])

    weight_backward = dense_weight_backward(associated_forward = forward_operation)
    weight_graph = operational_element(weight_backward, tags = ['Backward'])


    weight_update = update_operation(0.01, forward_operation)
    update_graph = operational_element(weight_update, tags = ['Update'])
    update_graph.add_input(weight_graph)


    self._bwd = backward_graph
    self._bwd_w = weight_graph
    self._fwd = forward_graph

    if previous_element.has_back:
      previous_element.connect_back(self)

  def connect_back(self, previous_element):
    backward_graph_input = previous_element.get_backward_output()

    self._bwd.add_input(backward_graph_input)
    self._bwd_w.add_input(backward_graph_input)

  def __repr__(self):
    if self._fwd._called_setup is False:
      self._fwd.setup(tag = 'Forward')
    self._fwd.forward(tag = 'Forward')    
    return self._fwd.__repr__()

  def forward(self):
    self._fwd.forward()


  @property
  def weights(self):
    return self._fwd._op.get_weights_signature()

  def get_forward_output(self): return self._fwd

  def get_backward_output(self): return self._bwd
