import numpy as np
import renom as rm
from renom.graph.core import UserGraph, GraphMultiStorage, operation, GraphFactory
import renom.utility.initializer as init


class concatenate_forward(operation):

    name = 'Concatenate (F)'

    def __init__(self, axis=None):
        self.axis = axis

    def setup(self, inputs):
        assert isinstance(inputs, (list, tuple)), \
            "Concatenate accepts only list or tuple of array."
        inputs = [a['y'] for a in inputs]
        self._inputs = inputs
        gpus = inputs[0].gpus
        axis = self.axis

        out_shape = inputs[0].shape[:axis] + \
            (int(np.sum([int(a.shape[axis]) for a in inputs])), ) + inputs[0].shape[axis + 1:]

        self.gpus = gpus
        outs = GraphMultiStorage(shape=out_shape, gpus=gpus)
        self._split_index = np.cumsum([a.shape[axis] for a in inputs[:-1]]).tolist()
        self._outputs = outs
        self._vars = {'y': outs}

    def perform(self):
        if self.axis == 0:
            val = np.sum(a.shape[0].value for a in self._inputs)
            self._outputs.shape[0].value = val
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cuconcat([a[gpu] for a in self._inputs], self._outputs[gpu], axis=self.axis)


class concatenate_forward_cpu(concatenate_forward):

    def perform(self):
        ret = np.concatenate([a['cpu'] for a in self._inputs], axis=self.axis)
        self._outputs['cpu'] = ret


class concatenate_backward(operation):

    name = 'Concatenate (B)'

    def __init__(self, associated_forward, nth_input):
        self._fwd_op = associated_forward
        self._nth_input = nth_input

    def setup(self, inputs):
        n = self._nth_input
        inputs = inputs[0]['dy']
        gpus = inputs.gpus
        fwd_inputs = self._fwd_op._inputs[n]
        out_shape = fwd_inputs.shape
        outs = GraphMultiStorage(shape=out_shape, gpus=gpus)

        self.gpus = gpus
        self._inputs = inputs
        self._outputs = outs
        self._split_index = self._fwd_op._split_index
        self.axis = self._fwd_op.axis
        self._vars = {'y': outs, 'dy': outs, id(fwd_inputs): outs}

    def perform(self):
        n = self._nth_input
        axis = self.axis
        sp_index = self._split_index
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            splitted = self._inputs[gpu].split(sp_index, axis=axis)
            self._outputs[gpu] = splitted[n]


class concatenate_backward_cpu(concatenate_backward):

    def perform(self):
        n = self._nth_input
        axis = self.axis
        sp_index = self._split_index
        splitted = np.split(self._inputs['cpu'], sp_index, axis=axis)
        self._outputs['cpu'] = splitted[n]


class ConcatenateElement(UserGraph):

    name = 'Concatenate'

    def __init__(self, previous_elements=None, axis=0):
        fwd_op = concatenate_forward(axis=axis) if rm.is_cuda_active() \
            else concatenate_forward_cpu(axis=axis)
        bwd_ops = [concatenate_backward(fwd_op, nth) if rm.is_cuda_active()
                   else concatenate_backward_cpu(fwd_op, nth) for nth in range(len(previous_elements))]
        super().__init__(fwd_op, bwd_ops, previous_elements)


class ConcatenateGraphElement(GraphFactory):

    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def connect(self, other):
        assert isinstance(other, (list, tuple)), \
            "Concatenate accepts only list or tuple of array."
        return ConcatenateElement(other, axis=self.axis)


def concatenate(self, axis=0):
    assert isinstance(self, (list, tuple)), \
        "Concatenate accepts only list or tuple of array."
    return ConcatenateElement([*self], axis=axis)
