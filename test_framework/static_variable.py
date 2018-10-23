import numpy as np
import renom as rm
from new_gpu import multi_gpu_variable
from graph_element import graph_element

class static_variable(graph_element):

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
    self._memory_info= multi_gpu_variable(shape = shape, gpus=self._num_gpus, allocate_backward=False)
    self.shape = shape
    
    for gpu, handle in enumerate(rm.cuda.RenomHandlers(self._num_gpus)):
      self._memory_info[gpu].to_gpu(value[gpu]) 

  def forward(self): pass  
