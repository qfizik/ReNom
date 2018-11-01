import numpy as np
import renom as rm
from .core import multi_gpu_variable, operational_element, learnable_graph_element, operation

class static_value(operation):

  name = 'Static Variable'

  def __init__(self, value):
    self._outputs = value
    self._vars = { 'y' : self._outputs }

  def setup(self, inputs, storage): pass
  def perform(self): pass


class StaticVariableElement(learnable_graph_element):

  has_back = False

  def __init__(self, value, num_gpus = 1):
    val = multi_gpu_variable(shape = value.shape, gpus = num_gpus)
    for gpuv in val:
      gpuv.to_gpu(value)
    self._forward_operations = [ static_value(val) ]
    super().__init__()

