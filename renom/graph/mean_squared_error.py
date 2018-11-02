import renom as rm
from .core import learnable_graph_element, operation, operational_element, multi_gpu_variable

class mean_squared_forward(operation):

  name = 'Mean Squared (F)'

  def setup(self, inputs, storage):
    predictions = inputs[0]['y']
    real_values = inputs[1]['y']
    self._graph_input = predictions
    self._label_input = real_values

    assert predictions.shape == real_values.shape
    output = multi_gpu_variable(shape = predictions.shape, gpus = predictions._num_gpus)

    self._outputs = output

  def perform(self): pass

  def get_output_signature(self): raise AttributeError

  def __repr__(self): return self._outputs.__repr__()
  

class mean_squared_backward(operation):

  name = 'Mean Squared (B)'

  def setup(self, inputs, storage):
  
    predictions = inputs[0]
    real_values = inputs[1]
    self._graph_input = predictions
    self._label_input = real_values
    output = multi_gpu_variable(shape = predictions.shape, gpus = predictions._num_gpus)
    self._outputs = output
    
  def perform(self):
    with rm.cuda.RenomHandler() as handle:
      rm.cuda.cusub(self._graph_input[0], self._label_input[0], self._outputs[0], handle)
      rm.cuda.cumul(self._outputs[0], 2, self._outputs[0], handle)

  def get_output_signature(self): return self._outputs


class MeanSquaredElement(learnable_graph_element):

  def __init__(self, previous_elements = None):

    self._forward_operations = [ mean_squared_forward() ]
    self._backward_operations = [ mean_squared_backward() ] 
    super().__init__(previous_elements = previous_elements)

