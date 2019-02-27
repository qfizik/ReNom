import renom as rm
from renom.graph.core import UserGraph, GraphMultiStorage, operation, GraphFactory
import renom.utility.initializer as init
import numpy as np


class sum_forward(operation):

    name = 'Sum (F)'

    def __init__(self, axis=None, keepdims=True):
        self.axis = axis
        self.keepdims = keepdims

    def setup(self, inputs):
        inputs = inputs[0]['y']
        self._inputs = inputs
        gpus = inputs.gpus
        self.gpus = gpus
        if self.axis is None and not self.keepdims:
            out_shape = (1, )
        else:
            out_shape = np.sum(np.zeros(inputs.shape, dtype=np.bool),
                               axis=self.axis, keepdims=self.keepdims).shape
            if not out_shape:
                out_shape = (1, )
        outs = GraphMultiStorage(shape=out_shape, gpus=gpus)
        self._outputs = outs
        self._vars = {'y': outs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            r = rm.cuda.cusum(self._inputs[gpu], handle, axis=self.axis, keepdims=self.keepdims)
            self._outputs[gpu].copy_from(r)


class sum_forward_cpu(sum_forward):

    def perform(self):
        ret = np.sum(self._inputs['cpu'], axis=self.axis, keepdims=self.keepdims)
        self._outputs['cpu'] = ret


class sum_backward(operation):

    name = 'Sum (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        inputs = inputs[0]['dy']
        gpus = inputs.gpus
        out_shape = self._fwd_op._inputs.shape
        outs = GraphMultiStorage(shape=out_shape, gpus=gpus)
        fwd_inputs = self._fwd_op._inputs

        self.gpus = gpus
        self._inputs = inputs
        self._outputs = outs
        self._fwd_inputs = fwd_inputs
        self.axis = self._fwd_op.axis
        self.keepdims = self._fwd_op.keepdims
        axis = [self.axis] if isinstance(self.axis, (int, type(None))) else self.axis
        self.expand_shape = tuple([1 if (i in axis or axis[0] is None)
                                   else s for i, s in enumerate(fwd_inputs.shape)])
        self._ones = GraphMultiStorage(
            shape=fwd_inputs.shape, gpus=gpus, initializer=init.Constant(1))
        self._vars = {'y': outs, 'dy': outs, id(fwd_inputs): outs}

    def perform(self):
        axis = self.axis
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            dy = self._inputs[gpu].reshape(self.expand_shape)
            ones = self._ones[gpu]
            self._outputs[gpu] = dy * ones
            if axis is None:
                self._outputs[gpu] = ones * dy
            else:
                if not self.keepdims:
                    dy = ones * dy.reshape(self.expand_shape)
                else:
                    dy = ones * dy
                self._outputs[gpu] = dy


class sum_backward_cpu(sum_backward):

    def perform(self):
        axis = self.axis
        dy = self._inputs['cpu'].reshape(self.expand_shape)
        ones = self._ones['cpu']
        if axis is None:
            self._outputs['cpu'] = ones * dy
        else:
            if not self.keepdims:
                dy = ones * dy.reshape(self.expand_shape)
            else:
                dy = ones * dy
            self._outputs['cpu'] = dy


class SumElement(UserGraph):

    name = 'Sum'

    def __init__(self, previous_elements=None, axis=None, keepdims=False):
        fwd_op = sum_forward(axis=axis, keepdims=keepdims) if rm.is_cuda_active(
        ) else sum_forward_cpu(axis=axis, keepdims=keepdims)
        bwd_ops = [sum_backward(fwd_op) if rm.is_cuda_active() else sum_backward_cpu(fwd_op)]
        super().__init__(fwd_op, bwd_ops, previous_elements)


class Sum(GraphFactory):

    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def connect(self, other):
        ret = SumElement(other, axis=self.axis, keepdims=self.keepdims)
        return ret


def sum(self, axis=None, keepdims=False):
    return SumElement([self], axis=axis, keepdims=keepdims)


UserGraph.sum = sum
