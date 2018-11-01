import numpy as np
import renom as rm
from .core import operational_element, learnable_graph_element, operation, multi_gpu_variable

class dispatch(operation):

  name = 'Data Dispatcher'

  def __init__(self, value, batch_size = 128, num_gpus = 1):
    self._value = value
    self._batch_num = 0
    self._batch_size = batch_size
    out_shape = [ batch_size ]
    out_shape.extend(value.shape[1:])
    self._num_gpus = num_gpus
    self._outputs = multi_gpu_variable(shape = out_shape, gpus = num_gpus)
    self._vars = { 'y' : self._outputs }

  def setup(self, inputs, storage):
    self._storage = storage

  def perform(self):
    #if self._batch_num * self._batch_size > len(self._value):
    #  raise StopIteration
    for gpu, handle in enumerate(rm.cuda.RenomHandlers(self._num_gpus)):
      cur_slice = slice(self._batch_num * self._batch_size, (1 + self._batch_num) * self._batch_size)
      arr = self._value[cur_slice]
      if len(arr) < self._batch_size:
        #self._outputs[gpu] = self._outputs[gpu].batch_slice(len(arr))
        raise StopIteration
      pin = handle.getPinnedMemory(arr)
      self._outputs[gpu].to_gpu(pin)
      self._batch_num += 1

class data_entry_element(learnable_graph_element):

  has_back = False

  def __init__(self, data_op, previous_element = None):
    self._forward_operations = [ data_op ]

    super().__init__(previous_elements = previous_element)

class DistributorElement:


  def __init__(self, data, labels, batch_size = 64, num_gpus = 1):
    super().__init__()
    self._data = data
    self._labels = labels
    self._batch_size = batch_size
    self._num_gpus = num_gpus

    data_op = dispatch(data, num_gpus = num_gpus)
    lbls_op = dispatch(labels, num_gpus = num_gpus)

    self._data_graph = data_entry_element(data_op)
    self._label_graph = data_entry_element(lbls_op)

  def forward(self): pass

  def get_output_graphs(self): return self._data_graph, self._label_graph

  def reset(self):
    self._data_graph._batch_num = 0
    self._label_graph._batch_num = 0

  def __repr__(self): return self._data_graph.__repr__()

