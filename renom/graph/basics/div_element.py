#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

import numpy as np

import renom as rm
from renom.graph.core import operation, GraphMultiStorage, operational_element, UserGraph, GraphFactory
from renom.graph.utils import broad_cast, cu_broad_cast
from renom.graph import populate_graph


class div_forward(operation):

    name = 'Div (F)'

    def __init__(self):
        self._a = None
        self._b = None

    def setup(self, inputs):
        a = inputs[0]['y']
        b = inputs[1]['y']
        assert len(a) == len(b)
        self.gpus = a.gpus
        self._a = a
        self._b = b
        output_shape = (np.zeros(a.shape) + np.zeros(b.shape)).shape
        self._c = GraphMultiStorage(shape=output_shape, gpus=self.gpus)
        self._vars = {'a': a, 'b': b, 'y': self._c}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cudiv(self._a[gpu], self._b[gpu], self._c[gpu], handle)


class div_forward_cpu(div_forward):

    def perform(self):
        a = self._a['cpu']
        b = self._b['cpu']
        self._c['cpu'] = a / b


class div_backward(operation):

    name = 'Div (B)'

    def __init__(self, associated_forward, key):
        self._fwd_op = associated_forward
        self._key = key

    def setup(self, inputs):
        self._inputs = inputs[0]['y']
        key = self._key
        key_value = self._fwd_op.get_key(key)
        gpus = key_value.gpus
        output_shape = key_value.shape
        outputs = GraphMultiStorage(shape=output_shape, gpus=gpus, initializer=None)

        if key == "a":
            self._opposit_value = self._fwd_op.get_key('b')
        elif key == "b":
            self._opposit_value = self._fwd_op.get_key('a')
        else:
            raise Exception()

        self.gpus = gpus
        self._fwd_in = key_value
        self._vars = {'y': outputs, 'dy': outputs, id(key_value): outputs}
        self._outputs = outputs

    def perform(self):
        for i, (gpu, handle) in enumerate(rm.cuda.RenomHandlers(self.gpus)):
            a = self._fwd_in[gpu]
            b = self._opposit_value[gpu]
            dy = self._inputs[gpu]
            if self._key == "a":
                dy = dy / b
            elif self._key == "b":
                dy = a**(-2.0) * -1.0 * b * dy

            if a.shape != dy.shape:
                dy = cu_broad_cast(a, dy)
            else:
                dy = dy
            self._outputs[gpu] = dy


class div_backward_cpu(div_backward):

    def perform(self):
        a = self._fwd_in['cpu']
        b = self._opposit_value['cpu']
        dy = self._inputs['cpu']
        if self._key == "a":
            dy = dy / b
        else:
            dy = a**(-2.0) * -1.0 * b * dy
        if a.shape == dy.shape:
            self._outputs['cpu'] = dy
        else:
            self._outputs['cpu'] = broad_cast(a, dy)


class DivElement(UserGraph):

    _name = 'Div Element'

    def __init__(self, previous_elements=None):

        fwd_op = div_forward() if rm.is_cuda_active() else div_forward_cpu()
        bwd_ops = [div_backward(fwd_op, 'a') if rm.is_cuda_active() else div_backward_cpu(fwd_op, 'a'),
                   div_backward(fwd_op, 'b') if rm.is_cuda_active() else div_backward_cpu(fwd_op, 'b')]
        super().__init__(fwd_op, bwd_ops, previous_elements)


@populate_graph
class Div(GraphFactory):

    def connect(self, lhs, rhs):
        return DivElement([lhs, rhs])


def _div(self, other):
    ret = DivElement([self, other])
    return ret


UserGraph.__div__ = _div
UserGraph.__idiv__ = _div
UserGraph.__rdiv__ = _div

UserGraph.__truediv__ = _div
UserGraph.__itruediv__ = _div
UserGraph.__rtruediv__ = _div
