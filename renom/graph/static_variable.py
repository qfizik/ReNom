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

  _has_back = False
  _name = 'Static Element'

  def __init__(self, value, num_gpus = 1):
    gpu_list = [gpu for gpu in range(num_gpus)]
    val = multi_gpu_variable(shape = value.shape, gpus = gpu_list)
    for gpuv in val:
      gpuv.to_gpu(value)
    self._value = val
    self._forward_operations = [ static_value(val) ]
    super().__init__()

  @property
  def value(self):
    return self._value
  
  @value.setter
  def value(self, val):
    assert isinstance(val, np.ndarray)    
    if self._value.shape == val.shape:
      for gpuv in self._value:
        gpuv.to_gpu(val)  
    else:
      #TODO: FIX THIS
      assert False
    

