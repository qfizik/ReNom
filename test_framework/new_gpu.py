import numpy as np
import renom as rm

class multi_gpu_variable:

  def __init__(self, shape, gpus = 1, allocate_backward = True, initializer = None):
    self._num_gpus = gpus
    self._forwards = []
    self._shape = shape
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu) as handle:
        if initializer is not None:
          arr = initializer(shape)
        else:
          arr = np.ones(shape)
        self._forwards.append(rm.GPUValue(array=arr, shape=shape))


  def get_forwards(self, gpu_id):
    return self._forwards[gpu_id]

  def __iter__(self):
    for _fwd in self._forwards:
      yield _fwd

  def __len__(self):
    return self._num_gpus

  def __getitem__(self, index):
    return self._forwards[index]
      



