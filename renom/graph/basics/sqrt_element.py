from renom.graph.core import operation, operational_element, UserGraph, GraphMultiStorage, GraphFactory
import renom as rm
import numpy as np


class sqrt_forward(operation):

    name = 'Sqrt (F)'

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
            rm.cuda.cusqrt(self._inputs[gpu], self._outputs[gpu])


class sqrt_forward_cpu(sqrt_forward):

    def perform(self):
        self._outputs['cpu'] = np.sqrt(self._inputs['cpu'])


class sqrt_backward(operation):

    name = 'Sqrt (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        inputs = inputs[0]['y']
        fwd_inputs = self._fwd_op._inputs
        shape = fwd_inputs.shape
        gpus = fwd_inputs.gpus

        self.gpus = gpus
        self._inputs = inputs
        self._fwd_outputs = self._fwd_op._outputs
        self._outputs = GraphMultiStorage(shape=shape, gpus=gpus)
        self._vars = {'y': self._outputs, 'dy': self._outputs, id(fwd_inputs): self._outputs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            self._outputs[gpu] = 0.5 / self._fwd_outputs[gpu] * self._inputs[gpu]


class sqrt_backward_cpu(sqrt_backward):

    def perform(self):
        self._outputs['cpu'] = 0.5 / self._fwd_outputs['cpu'] * self._inputs['cpu']


class SqrtElement(UserGraph):

    _name = 'Sqrt Element'

    def __init__(self, previous_element=None):
        fwd_op = sqrt_forward() if rm.is_cuda_active() else sqrt_forward_cpu()
        bwd_ops = [sqrt_backward(fwd_op) if rm.is_cuda_active() else sqrt_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops,
                         previous_elements=previous_element)


def sqrt(self):
    ret = SqrtElement([self])
    return ret


UserGraph.sqrt = sqrt
