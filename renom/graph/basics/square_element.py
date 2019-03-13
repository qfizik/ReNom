from renom.graph.core import operation, operational_element, UserGraph, GraphMultiStorage, GraphFactory
import renom as rm
import numpy as np


class square_forward(operation):

    name = 'Square (F)'

    def setup(self, inputs):
        inputs = inputs[0]['y']
        output_shape = inputs.shape
        gpus = inputs.gpus

        self._inputs = inputs
        self.gpus = gpus
        self._outputs = GraphMultiStorage(shape=output_shape, gpus=gpus)
        self._vars = {'y': self._outputs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            self._outputs[gpu] = self._inputs[gpu] * self._inputs[gpu]


class square_forward_cpu(square_forward):

    def perform(self):
        self._outputs['cpu'] = self._inputs['cpu'] * self._inputs['cpu']


class square_backward(operation):

    name = 'Square (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        inputs = inputs[0]['y']
        fwd_inputs = self._fwd_op._inputs
        shape = fwd_inputs.shape
        gpus = fwd_inputs.gpus

        self.gpus = gpus
        self._inputs = inputs
        self._fwd_inputs = fwd_inputs
        self._outputs = GraphMultiStorage(shape=shape, gpus=gpus)
        self._vars = {'y': self._outputs, 'dy': self._outputs, id(fwd_inputs): self._outputs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            self._outputs[gpu] = self._fwd_inputs[gpu] * 2 * self._inputs[gpu]


class square_backward_cpu(square_backward):

    def perform(self):
        self._outputs['cpu'] = self._fwd_inputs['cpu'] * 2 * self._inputs['cpu']


class SquareElement(UserGraph):

    _name = 'Square Element'

    def __init__(self, previous_element=None):
        fwd_op = square_forward() if rm.is_cuda_active() else square_forward_cpu()
        bwd_ops = [square_backward(fwd_op) if rm.is_cuda_active() else square_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops,
                         previous_elements=previous_element)


def square(self):
    ret = SquareElement([self])
    return ret


UserGraph.square = square
