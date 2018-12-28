import renom as rm
from renom.graph.core import UserGraph, GraphMultiStorage, operation, GraphFactory
import renom.utility.initializer as init
import numpy as np


class sum_forward(operation):

    name = 'Sum (F)'

    def setup(self, inputs):
        inputs = inputs[0]['y']
        self._inputs = inputs
        gpus = inputs.gpus
        self.gpus = gpus
        out_shape = (1, )
        outs = GraphMultiStorage(shape=out_shape, gpus=gpus)
        self._outputs = outs
        self._vars = {'y': outs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            r = rm.cuda.cusum(self._inputs[gpu], handle)
            self._outputs[gpu].copy_from(r)


class sum_forward_cpu(sum_forward):

    def perform(self):
        ret = np.sum(self._inputs['cpu']).reshape(1)
        self._outputs['cpu'] = ret


class sum_backward(operation):

    name = 'Sum (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        inputs = inputs[0]['y']
        self._inputs = inputs
        gpus = inputs.gpus
        self.gpus = gpus
        out_shape = self._fwd_op._inputs.shape
        outs = GraphMultiStorage(shape=out_shape, gpus=gpus)
        ones = GraphMultiStorage(shape=out_shape, gpus=gpus, initializer=init.Constant(1))
        self._outputs = outs
        self._ones = ones
        self._vars = {'y': outs, id(self._fwd_op._inputs): outs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cumul(self._inputs[gpu], self._ones[gpu], self._outputs[gpu], handle)


class sum_backward_cpu(sum_backward):

    def perform(self):
        self._outputs['cpu'] = self._inputs['cpu'] * self._ones['cpu']


class SumElement(UserGraph):

    name = 'Sum'

    def __init__(self, previous_elements=None):
        fwd_op = sum_forward() if rm.is_cuda_active() else sum_forward_cpu()
        bwd_ops = [sum_backward(fwd_op) if rm.is_cuda_active() else sum_backward_cpu(fwd_op)]
        super().__init__(fwd_op, bwd_ops, previous_elements)


class SumGraphElement(GraphFactory):

    def connect(self, other):
        ret = SumElement(other)
        return ret
