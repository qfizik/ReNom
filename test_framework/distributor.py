import numpy as np
import renom as rm


class distributor:

  def __init__(self, data, labels, batch_size = 64, num_gpus = 1):
    self._data = data
    self._labels = labels
    self._batch_size = batch_size
    self._num_gpus = num_gpus
    self._cur_batch = 0

  def setup(self):
    output_shape = [self._batch_size]
    output_shape.extend(self._data.shape[1:])
    label_shape = [self._batch_size]
    label_shape.extend(self._labels.shape[1:])
    self._outputs = []
    self._label_outputs = []
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu):
        self._outputs.append(rm.GPUValue(shape=output_shape))
        self._label_outputs.append(rm.GPUValue(shape=label_shape))
  

  def forward(self):
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu) as handle:
        pin = handle.getPinnedMemory(self._data[self._cur_batch * self._batch_size : (self._cur_batch + 1) * self._batch_size])
        if pin.shape[0] < self._batch_size:
          raise StopIteration
        self._outputs[gpu].to_gpu(pin)
        pin = handle.getPinnedMemory(self._labels[self._cur_batch * self._batch_size : (self._cur_batch + 1) * self._batch_size])
        self._label_outputs[gpu].to_gpu(pin)
      self._cur_batch += 1
