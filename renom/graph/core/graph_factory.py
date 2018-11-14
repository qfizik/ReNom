import renom as rm
import abc
import numpy as np
from .learnable_graph import learnable_graph_element
from .operation import operation
from .new_gpu import multi_gpu_variable

class variable_input(operation):

  name = 'Variable'

  def __init__(self):
    val = multi_gpu_variable()
    self._vars = { 'y' : val }

  def setup(self, inputs, storage): pass

  def perform(self): pass

class graph_variable(learnable_graph_element):

  def __init__(self):
    fwd_op = variable_input()
    bwd_ops = [ ]
    super().__init__(forward_operation = fwd_op, backward_operations = bwd_ops)
  


class GraphFactory(abc.ABC):

  @abc.abstractmethod
  def connect(self, other): pass
  
  def __call__(self, *other):
    return self.connect(*other)
