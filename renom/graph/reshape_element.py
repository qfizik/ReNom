from .core import operation, operational_element, learnable_graph_element, multi_gpu_variable
import renom as rm
import numpy as np

class reshape_op(operation):

  def __init__(self, shape):
    self._new_shape = shape

  def setup(self, inputs, storage):
    self._inputs = inputs[0]['y']
    new_shape = [ self._inputs.shape[0] ]
    new_shape.extend(self._new_shape)
    new_shape = np.empty(self._inputs.shape).reshape(new_shape).shape
    gpus = self._inputs.gpus
    self._outputs = multi_gpu_variable(shape = new_shape, gpus = gpus, ptrs = self._inputs)
    self._vars = {'y' : self._outputs }

  def perform(self): pass



class reshape_op_back(operation):

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):
    self._inputs = inputs[0]['y']
    shape = self._fwd_op._inputs.shape
    gpus = self._inputs.gpus
    self._outputs = multi_gpu_variable(shape = shape, gpus = gpus, ptrs = self._inputs)

  def perform(self): pass

  def get_output_signature(self): return self._outputs

  def __repr__(self): return self._outputs.__repr__()

class ReshapeElement(learnable_graph_element):

  has_back = True

  def __init__(self, shape, previous_element = None):
    self._shape = shape
    fwd_op = reshape_op(shape)
    self._forward_operations = [ fwd_op ]
    self._backward_operations = [ reshape_op_back(fwd_op) ]
    super().__init__(previous_elements = previous_element)
