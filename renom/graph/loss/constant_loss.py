import renom as rm
from renom.graph.core import operation, operational_element, learnable_graph_element, multi_gpu_variable, GraphFactory
from renom.graph.function.sum_element import sum_forward
import renom.utility.initializer as init 


class constant_loss_backward(operation):
  
  name = 'Constant (B)'

  def setup(self, inputs, storage):
  
    inputs = inputs[0]['y']
    gpus = inputs.gpus
    outputs = multi_gpu_variable(shape = inputs.shape, gpus = gpus, initializer = init.Constant(1))
    self._outputs = outputs
    self._vars = { 'y' : outputs , 'dy' : outputs}
    self.ready = True

  def perform(self): pass

class ConstantLoss(learnable_graph_element):
  
  is_connector_element = True

  def __init__(self, previous_element = None):
    super().__init__(forward_operation = sum_forward(), backward_operations = [ constant_loss_backward()], previous_elements = previous_element)
    self._bwd_graphs[0].add_input(previous_element.get_forward_output())
    self._bwd_graphs[0].add_input(self._fwd)


    
class ConstantLossElement(GraphFactory):

  def connect(self, other):
    return ConstantLoss(previous_element = other)
