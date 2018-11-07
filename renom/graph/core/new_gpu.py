import numpy as np
import renom as rm

class shared_val:
  def __new__(cls, val, m = None):
    return super().__new__(cls)
  def __init__(self, val):
    self._val = val
    self._max = val
  @property
  def value(self):
    return self._val
  @value.setter
  def value(self, val):
    if val <= self._max:
      self._val = val
    else:
      raise AttributeError('Setting val above initial value')

  def __int__(self): return self._val
  def __float__(self): return float(self._val)
  def __mul__(self, other): return self._val * int(other)
  def __add__(self, other): return self._val + int(other)
  def __eq__(self, other): return self._val == int(other)
  def __index__(self): return self._val
  def __repr__(self): return self._val.__repr__()

class multi_gpu_variable:

  def __init__(self, shape = None, gpus = [ 0 ], initializer = None, ptrs = None):
    assert isinstance(gpus, list)
    self.gpus = gpus
    self._gpuvalues = {}
    self._initializer = initializer
    self._finished_setup = False
    self._ptrs = ptrs
    self.shape = shape
    if ptrs is not None:
      assert isinstance(ptrs, multi_gpu_variable)
      shp = list(self.shape)
      shp[0] = ptrs.shape[0]
      self.shape = shp
    self._create_values()

  def _create_values(self):
    for gpu in self.gpus:
      with rm.cuda.RenomHandler(gpu) as handle:
        if self._initializer is not None:
          arr = self._initializer(self.shape)
        else:
          arr = np.ones(self.shape)
        self[gpu] = rm.GPUValue(array=arr, shape=self.shape, ptr = self._ptrs[gpu]._ptr if self._ptrs is not None else None)

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

  @property
  def gpus(self):
    return self._gpus

  @gpus.setter
  def gpus(self, val):
    assert isinstance(val, list)
    self._gpus = val


  def __iter__(self):
    for key in self._gpuvalues.keys():
      yield self[key]

  def __len__(self):
    return len(self._gpus)

  def __getitem__(self, index):
    return self._gpuvalues[index]

  def __setitem__(self, index, value):
    self._gpuvalues[index] = value
      
  def __repr__(self):
    assert self._finished_setup is True
    return self._gpuvalues[0].new_array().__repr__()

  def as_ndarray(self): return self._gpuvalues[0].new_array()
