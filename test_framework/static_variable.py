import numpy as np
import renom as rm
from new_gpu import multi_gpu_variable
from graph_element import graph_element, operational_element
from learnable_graph import learnable_graph_element
from operation import operation

class static_value(operation):

  name = 'Static Variable'

  def __init__(self, value):
    self._outputs = multi_gpu_variable(shape = value.shape, gpus = 1)
    self._outputs[0].to_gpu(value)

  def setup(self, inputs): pass

  def perform(self): pass

  def get_output_signature(self): return self._outputs

  def __repr__(self): return self._outputs.__repr__()

class static_variable(learnable_graph_element):

  has_back = False

  def __init__(self, value):
    super().__init__()

    if isinstance(value, np.ndarray):
      shape = value.shape
      value = [ value ]
    elif isinstance(value, list):
      shape = value[0].shape
      for _v in value:
        assert isinstance(_v, np.ndarray)
        assert _v.shape == shape 

    self._num_gpus = len(value)
    self._gpus = [gpu for gpu in range(self._num_gpus)]
    self.shape = shape
    
    op = static_value(value[0])

    self._fwd = operational_element(op)



  def forward(self): pass  

  def __repr__(self):
    return self._fwd.__repr__()

  def get_forward_output(self): return self._fwd

  def get_backward_output(self): return None


