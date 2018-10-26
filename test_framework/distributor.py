import numpy as np
import renom as rm
from graph_element import operational_element
from learnable_graph import learnable_graph_element
from operation import operation
from new_gpu import multi_gpu_variable

class dispatch(operation):

  name = 'Data Dispatcher'

  def __init__(self, value, batch_size = 128):
    self._value = value
    self._batch_num = 0
    self._batch_size = batch_size
    out_shape = ( batch_size , value.shape[1] )
    self._outputs = multi_gpu_variable(shape = out_shape, gpus = 1)

  def setup(self, inputs): pass

  def perform(self):
    with rm.cuda.RenomHandler() as handle:
      cur_slice = slice(self._batch_num * self._batch_size, (1 + self._batch_num) * self._batch_size)
      arr = self._value[cur_slice]
      if len(arr) < self._batch_size:
        raise StopIteration
      pin = handle.getPinnedMemory(arr)
      self._outputs[0].to_gpu(pin)
      self._batch_num += 1

  def get_output_signature(self): return self._outputs

class distributor(learnable_graph_element):

  has_back = False

  def __init__(self, data, labels, batch_size = 64, num_gpus = 1):
    super(distributor, self).__init__()
    self._data = data
    self._labels = labels
    self._batch_size = batch_size
    self._num_gpus = num_gpus

    data_op = dispatch(data)
    lbls_op = dispatch(labels)

    data_graph = operational_element(data_op)
    label_graph = operational_element(lbls_op)

    self._data_graph = data_graph
    self._label_graph = label_graph

  def forward(self): pass

  def get_forward_output(self): return self._data_graph

  def get_labels_graph(self): return self._label_graph

  def reset(self):
    self._data_graph._batch_num = 0
    self._label_graph._batch_num = 0

  def __repr__(self): return self._data_graph.__repr__()

