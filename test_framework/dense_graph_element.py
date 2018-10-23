from learnable_graph import learnable_graph_element
from operation import operation
from new_gpu import multi_gpu_variable
import renom.utility.initializer as init
import renom as rm

class dense_forward(operation):

  def __init__(self, output_size):
    
    self._output_size = output_size
    self._output = None

  def setup(self, inputs):
    
    self._init = init.GlorotNormal()

    self._inputs = inputs
    weight_shape = ( inputs[0].shape[1] , self._output_size )
    weights = multi_gpu_variable( shape = weight_shape , gpus = 1, initializer = self._init)
    output_shape = ( inputs[0].shape[0] , self._output_size )
    outputs = multi_gpu_variable( shape = output_shape, gpus = 1, initializer = self._init)

    self._weights = weights
    self._outputs = outputs 

  def perform(self):
    with rm.cuda.RenomHandler() as handle:
      rm.cuda.cublas_gemm(self._inputs[0], 0, self._weights[0], 0, self._outputs[0], handle)

  def get_output_signature(self): return self._outputs

  def get_weights_signature(self): return self._weights

class dense_backward(operation):

  def __init__(self): pass


  def setup(self, inputs, weights, output_size):

    self._inputs = inputs
    self._weights = weights

    output_shape = ( inputs[0].shape[0] , output_size )
    outputs = multi_gpu_variable( shape = output_shape , gpus = 1)

    self._outputs = outputs

  def perform(self):
    with rm.cuda.RenomHandler as handle:
      rm.cuda.cublas_gemm(self._inputs[0], 0, self._weights[0], 1, self._outputs[0], handle)

  def get_output_signature(self): return self._outputs


class dense_graph_element(learnable_graph_element):

  def __init__(self, output_size, previous_element):
    
    self._output_size = output_size
    forward_operation = dense_forward(output_size)
    forward_graph =  operational_element(forward_operation)

    prev_graph_input = previous_element.get_output()
    forward_graph.set_input(prev_graph_input)

    backward_operation = dense_backward()
    backward_graph = operational_element(backward_operation)

    prev_graph_back_input = previous_element.get_backward_output()
    prev_graph_back_input.set_input(backward_graph)

    self._fwd = forward_graph
    self._bwd = backward_graph
