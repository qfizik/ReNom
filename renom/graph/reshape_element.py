from operation import operation
from graph_element import operational_element
from learnable_graph import learnable_graph_element
from new_gpu import multi_gpu_variable
import renom as rm

class reshape_op(operation):

  def __init__(self, shape):
    self._new_shape = shape

  def setup(self, inputs):
    self._inputs = inputs[0]
    shape = [self._inputs.get_shape()[0]]
    shape.extend(self._new_shape)
    self._outputs = multi_gpu_variable(shape = shape, ptrs = self._inputs)

  def perform(self): pass

  def get_output_signature(self): return self._outputs

  def __repr__(self): return self._outputs.__repr__()


class reshape_op_back(operation):

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs):
    self._inputs = inputs[0]
    shape = self._fwd_op._inputs.get_shape()
    self._outputs = multi_gpu_variable(shape = shape, ptrs = self._inputs)

  def perform(self): pass

  def get_output_signature(self): return self._outputs

  def __repr__(self): return self._outputs.__repr__()

class reshape_element(learnable_graph_element):

  has_back = True

  def __init__(self, shape, previous_element = None):
    self._shape = shape
    super().__init__(previous_elements = previous_element)

  def connect(self, previous_element):
  
    forward_operation = reshape_op(self._shape)
    forward_graph = operational_element(forward_operation)

    backward_operation = reshape_op_back(forward_operation)
    backward_graph = operational_element(backward_operation)

    prev_graph_input = previous_element.get_forward_output()
    forward_graph.add_input(prev_graph_input)

    self._fwd = forward_graph
    self._bwd = backward_graph

    if previous_element.has_back:
      previous_element.connect_back(self)

  
  def connect_back(self, previous_element):
    backward_graph_input = previous_element.get_backward_output()

    self._bwd.add_input(backward_graph_input)

  def forward(self): pass
  def __repr__(self): return self._fwd.__repr__()
  def get_forward_output(self): return self._fwd
  def get_backward_output(self): return self._bwd
