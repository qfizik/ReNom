import renom as rm
from renom.layers.function.utils import im2col, col2im, imnpool, poolnim
from renom.graph.core import operation, UserGraph, GraphMultiStorage, GraphFactory
import numpy as np


class MaxUnPoolElement(UserGraph):

    def __init__(self, prev_pool, previous_element=None):
        fwd_op = unpool_forward(prev_pool) if rm.is_cuda_active() else unpool_forward_cpu(prev_pool)
        bwd_ops = [unpool_backward(fwd_op) if rm.is_cuda_active() else unpool_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_element)


class MaxUnPool(GraphFactory):

    def prepare(self, prev_pool):
        self._prev_pool = prev_pool._fwd._op

    def connect(self, other):
        ret = MaxUnPoolElement(self._prev_pool, previous_element=[other])
        return ret


class unpool_forward(operation):

    name = 'UnPool (F)'

    def __init__(self, prev_pool):
        self._prev_pool = prev_pool

    def setup(self, inputs):

        inputs = inputs[0]['y']

        self._inputs = inputs
        prev_pool = self._prev_pool

        out_shape = prev_pool._inputs.shape

        self.gpus = inputs.gpus
        self._prev_in = prev_pool._inputs
        outs = GraphMultiStorage(shape=out_shape, gpus=self.gpus)
        self._outputs = outs
        self._vars = {'y': outs}
        if rm.is_cuda_active():
            pd = prev_pool._pool_desc
            self._pool_desc = pd

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            x = self._inputs[gpu]
            px = self._prev_in[gpu]
            dx = self._outputs[gpu]
            rm.cuda.cuPoolingBackward(handle, self._pool_desc, px, x, x, dx)


class unpool_forward_cpu(unpool_forward):

    def perform(self):
        x = self._inputs['cpu']
        px = self._prev_in['cpu']
        ret = poolnim(px, x, self._prev_pool._kernel, self._prev_pool._stride,
                      self._prev_pool._padding, mode='max')
        self._outputs['cpu'] = ret


class unpool_backward(operation):

    name = 'UnPool (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):

        inputs = inputs[0]['y']
        self._inputs = inputs
        out_shape = self._fwd_op._inputs.shape
        self._fwd_in = self._fwd_op._inputs
        self._fwd_out = self._fwd_op._outputs
        self.gpus = inputs.gpus
        outs = GraphMultiStorage(shape=out_shape, gpus=self.gpus)
        self._outputs = outs
        self._vars = {'y': outs, id(self._fwd_in): outs}
        self._prev_pool = self._fwd_op._prev_pool

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            dy = self._inputs[gpu].new_array()
            px = self._prev_pool._inputs[gpu].new_array()

            dx = imnpool(px, self._prev_pool._kernel, self._prev_pool._stride,
                         self._prev_pool._padding, mode='max', alternate_input=dy)

            self._outputs[gpu] = rm.GPUValue(dx)


class unpool_backward_cpu(unpool_backward):

    def perform(self):
        dy = self._inputs['cpu']
        px = self._prev_pool._inputs['cpu']

        dx = imnpool(px, self._prev_pool._kernel, self._prev_pool._stride,
                     self._prev_pool._padding, mode='max', alternate_input=dy)

        self._outputs['cpu'] = dx
