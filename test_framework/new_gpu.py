import numpy as np
import renom as rm

class multi_gpu_variable:

  def __init__(self, shape = None, gpus = 1, allocate_backward = True, initializer = None):
    self._num_gpus = gpus
    self._forwards = []
    self._shape = shape
    self._initializer = initializer
    self._finished_setup = False
    if shape is not None:
      self._create_values()

  def set_shape(self, shape):
    assert self._shape is None
    self._shape = shape
    self._create_values()


  def set_gpus(self, gpus):
    self._num_gpus = gpus


  def _create_values(self):
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu) as handle:
        if self._initializer is not None:
          arr = self._initializer(self._shape)
        else:
          arr = np.ones(self._shape)
        self._forwards.append(rm.GPUValue(array=arr, shape=self._shape))

    self._finished_setup = True
    
  def get_shape(self): return self._shape

  def get_forwards(self, gpu_id):
    return self._forwards[gpu_id]

  def __iter__(self):
    for _fwd in self._forwards:
      yield _fwd

  def __len__(self):
    return self._num_gpus

  def __getitem__(self, index):
    return self._forwards[index]
      
  def __repr__(self):
    assert self._finished_setup is True
    return self._forwards[0].new_array().__repr__()


