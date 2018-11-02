import numpy as np
import renom as rm

class shared_val(int):
  def __new__(cls, val):
    return int.__new__(cls, val)
  def __init__(self, val):
    self._val = val
  @property
  def value(self):
    return self._val
  @value.setter
  def value(self, val):
    self._val = val
  def __int__(self): return self._val


class multi_gpu_variable:

  def __init__(self, shape = None, gpus = 1, initializer = None, ptrs = None):
    self._num_gpus = gpus
    self._gpuvalues = []
    self._initializer = initializer
    self._finished_setup = False
    self._ptrs = ptrs
    self.shape = shape
    self._create_values()

  def _create_values(self):
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu) as handle:
        if self._initializer is not None:
          arr = self._initializer(self.shape)
        else:
          arr = np.ones(self.shape)
        self._gpuvalues.append(rm.GPUValue(array=arr, shape=self.shape, ptr = self._ptrs[gpu]._ptr if self._ptrs is not None else None))

    self._finished_setup = True
    
  @property
  def shape(self):
    return self._shape

  @shape.setter
  def shape(self, val):
    if not isinstance(val[0], shared_val):
      new_val = shared_val(val[0])
      l = list(val)
      l[0] = new_val
      val = tuple(l)
    assert isinstance(val[0], shared_val)
    self._shape = val


  def __iter__(self):
    for _fwd in self._gpuvalues:
      yield _fwd

  def __len__(self):
    return self._num_gpus

  def __getitem__(self, index):
    return self._gpuvalues[index]

  def __setitem__(self, index, value):
    self._gpuvalues[index] = value
      
  def __repr__(self):
    assert self._finished_setup is True
    return self._gpuvalues[0].new_array().__repr__()

  def as_ndarray(self): return self._gpuvalues[0].new_array()
