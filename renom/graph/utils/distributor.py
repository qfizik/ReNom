import numpy as np
import renom as rm
from renom.graph.core import operational_element, learnable_graph_element, operation, multi_gpu_variable

class dispatch(operation):

  name = 'Data Dispatcher'
  roles = [ 'input' ]

  def __init__(self, value, batch_size = 128, num_gpus = 1):
    self._value = value
    self._batch_num = 0
    self._batch_size = batch_size
    out_shape = [ batch_size ]
    out_shape.extend(value.shape[1:])
    self._num_gpus = num_gpus
    self.gpus = [gpu for gpu in range(num_gpus)]
    self._outputs = multi_gpu_variable(shape = out_shape, gpus = self.gpus)
    self._vars = { 'y' : self._outputs }
    self._finished = False
    self._perm = np.random.permutation(len(self._value))

  def setup(self, inputs, storage):
    self._storage = storage
    self._batch_vars = [v.shape[0] for v in self._outputs]

  @property
  def value(self):
    return self._value
  @value.setter
  def value(self, new_val):
    self._value = new_val

  def perform(self):
    if self._finished:
      raise StopIteration
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      #handle.wait()
      cur_slice = slice(self._batch_num * self._batch_size, (1 + self._batch_num) * self._batch_size)
      arr = self._value[self._perm[cur_slice]]
      self._outputs[gpu].shape[0].value = len(arr)
      assert self._outputs[gpu].shape == arr.shape
      if len(arr) < self._batch_size:
        self._finished = True
      pin = handle.getPinnedMemory(arr)
      assert pin.shape == self._outputs[gpu].shape
      self._outputs[gpu].to_gpu(pin)
      self._batch_num += 1

  def reset(self):
    self._batch_num = 0
    self._finished = False
    self._perm = np.random.permutation(len(self._value))

  def set_batch_size(self, batch_size):
    self._batch_size = batch_size

class data_entry_element(learnable_graph_element):

  has_back = False

  def __init__(self, data_op, previous_element = None):
    fwd_op = data_op
    super().__init__(forward_operation = fwd_op, previous_elements = previous_element)

class DistributorElement:


  def __init__(self, data, labels, batch_size = 64, num_gpus = 1):
    super().__init__()
    self._data = data
    self._labels = labels
    self._batch_size = batch_size
    self._num_gpus = num_gpus

    data_op = dispatch(data, num_gpus = num_gpus, batch_size = batch_size)
    lbls_op = dispatch(labels, num_gpus = num_gpus, batch_size = batch_size)

    self._dt_op = data_op
    self._lb_op = lbls_op
    self._data_graph = data_entry_element(data_op)
    self._label_graph = data_entry_element(lbls_op)

  def forward(self): pass

  def getOutputGraphs(self): return self._data_graph, self._label_graph

  def change_data(self, new_data): self._dt_op.value = new_data
  def change_label(self, new_label): self._lb_op.value = new_label

  def reset(self):
    self._dt_op.reset()
    self._lb_op.reset()

  def __repr__(self): return self._data_graph.__repr__()

