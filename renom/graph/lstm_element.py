import renom as rm
from .core import operational_element, operation, learnable_graph_element, multi_gpu_variable


class lstm_forward(operation):
  
  def __init__(self, output_size):
    self._output_size = output_size
    

  def setup(self, inputs, storage):
    pass
    
  def perform(self):
    pass

class lstm_backward(operation):

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):
    pass

  def perform(self):
    pass


class LstmElement(learnable_graph_element):

  def __init__(self, output_size):
    self._output_size = output_size
    fwd_op = lstm_forward(output_size)
    self._forward_operations = [ fwd_op ]
    self._backward_operations = [ lstm_backward(fwd_op) ]
    super().__init__()

  
