#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

import numpy as np

import renom as rm
from renom.graph.core import operation, GraphMultiStorage, operational_element, UserGraph
from renom.graph.utils import broad_cast, cu_broad_cast
from renom.graph import populate_graph


class pow_forward(operation):

    name = 'Pow (F)'

    def __init__(self):
        assert "This class needs to be fixed."
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
            rm.cuda.cupow(self._a[gpu], self._b[gpu], self._c[gpu], handle)


class pow_forward_cpu(pow_forward):

    def perform(self):
        a = self._a['cpu']
        b = self._b['cpu']
        self._c['cpu'] = a ** b


class pow_backward(operation):

    name = 'Pow (B)'

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

        a = self._fwd_op.get_key("a")
        b = self._fwd_op.get_key("b")
        c = self._fwd_op.get_key("y")
        self._a = a if key == "a" else b
        self._b = b if key == "a" else a
        self._c = c
        self.gpus = gpus
        self._vars = {'y': outputs, 'dy': outputs, id(key_value): outputs}
        self._outputs = outputs

    def perform(self):
        for i, (gpu, handle) in enumerate(rm.cuda.RenomHandlers(self.gpus)):
            a = self._a[gpu]
            b = self._b[gpu]
            c = self._c[gpu]
            dy = self._inputs[gpu]
            if self._key == "a":
                dy = dy * a.__pow__(b - 1) * b
            else:
                log_b = b.empty_like_me()
                rm.cuda.culoge(b, log_b)
                dy = log_b * c * dy

            if a.shape != dy.shape:
                dy = cu_broad_cast(a, dy)
            else:
                dy = dy
            self._outputs[gpu] = dy


class pow_backward_cpu(pow_backward):

    def perform(self):
        a = self._a['cpu']
        b = self._b['cpu']
        c = self._c['cpu']
        dy = self._inputs['cpu']
        if self._key == "a":
            dy = dy * a**(b - 1) * b
        else:
            dy = dy * c * np.log(b)
        if a.shape == dy.shape:
            self._outputs['cpu'] = dy
        else:
            self._outputs['cpu'] = broad_cast(a, dy)


class PowElement(UserGraph):

    _name = 'Pow Element'

    def __init__(self, previous_elements=None):

        fwd_op = pow_forward() if rm.is_cuda_active() else pow_forward_cpu()
        bwd_ops = [pow_backward(fwd_op, 'b') if rm.is_cuda_active() else pow_backward_cpu(fwd_op, 'b'),
                   pow_backward(fwd_op, 'a') if rm.is_cuda_active() else pow_backward_cpu(fwd_op, 'a')]
        super().__init__(fwd_op, bwd_ops, previous_elements)


def _pow(self, other):
    ret = PowElement([self, other])
    return ret


UserGraph.__pow__ = _pow
UserGraph.__ipow__ = _pow
UserGraph.__rpow__ = _pow
