import renom as rm
from .core import operation, operational_element, learnable_graph_element, multi_gpu_variable

class constant_loss_forward(operation):
  
  name = 'Constant (F)'

  def setup(self, inputs, storage): pass
  def perform(self): pass

class constant_loss_backward(operation):
  
  name = 'Constant (B)'

  def setup(self, inputs, storage):
  
    inputs = inputs[0]['y']
    gpus = inputs.gpus
    outputs = multi_gpu_variable(shape = inputs.shape, gpus = gpus)
    self._outputs = outputs
    self._vars = { 'y' : outputs , 'dy' : outputs}

  def perform(self): pass

class ConstantLossElement(learnable_graph_element):
  
  is_connector_element = True

  def __init__(self, previous_element = None):
    self._forward_operations = [ constant_loss_forward() ]
    self._backward_operations = [ constant_loss_backward() ]

    super().__init__(previous_elements = previous_element)
    

  def connect(self, *previous_elements):
    super().connect(*previous_elements)
    for elem in self._previous_elements:
      prev_graph_input = elem.get_forward_output()
      self._bwd_graphs[0].add_input(prev_graph_input)

  def connect_back(self): assert False
