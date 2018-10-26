import numpy as nppredictions
import renom as rm
from operation import operation
from learnable_graph import learnable_graph_element
from graph_element import operational_element
from new_gpu import multi_gpu_variable

class mean_squared_forward(operation):

  name = 'Mean Squared (F)'

  def setup(self, inputs):
    predictions = inputs[0]
    real_values = inputs[1]
    self._graph_input = predictions
    self._label_input = real_values

    assert predictions.get_shape() == real_values.get_shape()
    output = multi_gpu_variable(shape = predictions.get_shape(), gpus = predictions._num_gpus)

    self._outputs = output

  def perform(self): pass

  #def get_output_signature(self): raise AttributeError
  def get_output_signature(self): return None

  def __repr__(self): return self._outputs.__repr__()
  

class mean_squared_backward(operation):

  name = 'Mean Squared (B)'

  def __init__(self):
    self._setup = False

  def setup(self, inputs):
  
    predictions = inputs[0]
    real_values = inputs[1]
    self._graph_input = predictions
    self._label_input = real_values

    output = multi_gpu_variable(shape = predictions.get_shape(), gpus = predictions._num_gpus)

    self._outputs = output
    self._setup = True
    
  def perform(self):
    assert self._setup is True
    with rm.cuda.RenomHandler() as handle:
      rm.cuda.cusub(self._graph_input[0], self._label_input[0], self._outputs[0], handle)

  def get_output_signature(self): return self._outputs


class mean_squared_element(learnable_graph_element):

  def __init__(self, previous_elements = None):
    self._calls = {}
    super().__init__(previous_elements = previous_elements)


  def connect(self, previous_element, labels):

    forward_operation = mean_squared_forward()
    forward_graph = operational_element(forward_operation, tags = ['Forward'])

    prev_graph_inputA = previous_element.get_forward_output()
    prev_graph_inputB = labels.get_labels_graph()
    forward_graph.add_input(prev_graph_inputA)
    forward_graph.add_input(prev_graph_inputB)

    backward_operation = mean_squared_backward()
    backward_graph = operational_element(backward_operation, tags = ['Backward'])
    backward_graph.add_input(prev_graph_inputA)
    backward_graph.add_input(prev_graph_inputB)

    self._fwd = forward_graph
    self._bwd = backward_graph

    if previous_element.has_back:
      previous_element.connect_back(self)

  def __repr__(self):
    self._fwd.setup(tag = 'Forward')
    self._fwd.forward(tag = 'Forward')
    return self._fwd.__repr__()

  def forward(self):
    self._fwd.forward(tag = 'Forward')

  def backward(self):
    self._bwd.setup(tag = 'Backward')
    self._bwd.forward(tag = 'Backward')


  def update(self):
    if self._bwd.is_setup() is False:
      self._fwd._start_points[0].setup(tag = 'Update')
      self._fwd._start_points[1].setup(tag = 'Update')
      self._fwd._start_points[0].gather_calls(self._calls, tag = 'Update')
      self._fwd._start_points[1].gather_calls(self._calls, tag = 'Update')
    for depth in self._calls:
      for call in self._calls[depth]:
        call()
    #self._fwd._start_points[0].forward(tag = 'Update')

  def setup(self):
    self._fwd.setup(tag = 'Backward')

  def print_tree(self):
    assert False
    self._fwd._start_points[1].print_tree()

  def get_forward_output(self): return self._fwd
  def get_backward_output(self): return self._bwd
