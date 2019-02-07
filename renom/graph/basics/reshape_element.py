from renom.graph.core import operation, operational_element, UserGraph, GraphMultiStorage, GraphFactory
import renom as rm
import numpy as np


class reshape_forward(operation):

    name = 'Reshape (F)'

    def __init__(self, shape):
        self._new_shape = shape

    def setup(self, inputs):
        self._inputs = inputs[0]['y']
        new_shape = [self._inputs.shape[0]]
        new_shape.extend(self._new_shape)
        new_shape = np.empty(self._inputs.shape).reshape(new_shape).shape
        gpus = self._inputs.gpus
        self._outputs = GraphMultiStorage(shape=new_shape, gpus=gpus, ptrs=self._inputs)
        self._vars = {'y': self._outputs}

    def perform(self):
        pass


class reshape_backward(operation):

    name = 'Reshape (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        self._inputs = inputs[0]['y']
        shape = self._fwd_op._inputs.shape
        gpus = self._inputs.gpus
        self._outputs = GraphMultiStorage(shape=shape, gpus=gpus, ptrs=self._inputs)
        self._vars = {'y': self._outputs}

    def perform(self):
        pass

    def __repr__(self):
        return self._outputs.__repr__()


class ReshapeElement(UserGraph):

    def __init__(self, shape, previous_element=None):
        self._shape = shape
        fwd_op = reshape_forward(shape)
        bwd_ops = [reshape_backward(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_element)


class Reshape(GraphFactory):

    def __init__(self, shape):
        super().__init__()
        self.shp = shape

    def connect(self, other):
        ret = ReshapeElement(self.shp, other)
        return ret
