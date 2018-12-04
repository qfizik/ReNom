import numpy as np
import renom as rm
from renom.graph.core import multi_gpu_variable, operational_element, learnable_graph_element, operation

class static_value(operation):

  name = 'Static Variable'

  def __init__(self, value):
    self._outputs = value
    self._vars = { 'y' : self._outputs }

  def setup(self, inputs, storage): pass
  def perform(self): pass


class StaticVariable(learnable_graph_element):

  _has_back = False
  _name = 'Static Element'

  def __init__(self, value, num_gpus = 1):
    if rm.is_cuda_active():
      gpu_list = [gpu for gpu in range(num_gpus)]
    else:
      gpu_list = 'cpu'
    val = multi_gpu_variable(shape = value.shape, gpus = gpu_list)
    if rm.is_cuda_active():
      for gpuv in val:
        gpuv.to_gpu(value)
    else:
      val['cpu'] = value
    self._value = val
    fwd_op = static_value(val)
    super().__init__(forward_operation = fwd_op)

  @property
  def value(self):
    return self._fwd._op.get_key('y')
  
  @value.setter
  def value(self, val):
    assert isinstance(val, np.ndarray)    
    if self._value.shape == val.shape:
      if rm.is_cuda_active():
        for gpuv in self._value:
          gpuv.to_gpu(val)  
      else:
        self._value['cpu'] = val
    else:
      #TODO: FIX THIS
      raise NotImplementedError 
    
