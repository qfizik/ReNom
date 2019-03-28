#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

import numpy as np

import renom as rm
from renom.graph.core import UserGraph, GraphMultiStorage, operation, GraphFactory
from renom.graph.train import initializer as init
from renom.graph import populate_graph
from renom.graph.basics import populate_basics


class mean_forward(operation):

    name = 'Mean (F)'

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
            out_shape = np.mean(np.zeros(inputs.shape, dtype=np.bool),
                                axis=self.axis, keepdims=self.keepdims).shape
            if not out_shape:
                out_shape = (1, )
        outs = GraphMultiStorage(shape=out_shape, gpus=gpus)
        self._outputs = outs
        self._vars = {'y': outs, id(inputs): outs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            r = rm.cuda.cumean(self._inputs[gpu], handle, axis=self.axis, keepdims=self.keepdims)
            self._outputs[gpu].copy_from(r)


class mean_forward_cpu(mean_forward):

    def perform(self):
        ret = np.mean(self._inputs['cpu'], axis=self.axis, keepdims=self.keepdims)
        self._outputs['cpu'] = ret


class mean_backward(operation):

    name = 'Mean (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        inputs = inputs[0]['y']
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
        self.reduced_size = np.prod([s for i, s in enumerate(fwd_inputs.shape) if i in axis or self.axis is None],
                                    dtype=rm.precision)
        self._ones = GraphMultiStorage(
            shape=fwd_inputs.shape, gpus=gpus, initializer=init.Constant(1))
        self._vars = {'y': outs, id(fwd_inputs): outs}

    def perform(self):
        axis = self.axis
        rsize = self.reduced_size
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            dy = self._inputs[gpu].reshape(self.expand_shape)
            ones = self._ones[gpu]
            self._outputs[gpu] = dy * ones
            if axis is None:
                self._outputs[gpu] = ones * dy / rsize
            else:
                if not self.keepdims:
                    dy = ones * dy.reshape(self.expand_shape) / rsize
                else:
                    dy = ones * dy / rsize
                self._outputs[gpu] = dy


class mean_backward_cpu(mean_backward):

    def perform(self):
        axis = self.axis
        rsize = self.reduced_size
        dy = self._inputs['cpu'].reshape(self.expand_shape)
        ones = self._ones['cpu']
        if axis is None:
            self._outputs['cpu'] = ones * dy / rsize
        else:
            if not self.keepdims:
                dy = ones * dy.reshape(self.expand_shape) / rsize
            else:
                dy = ones * dy / rsize
            self._outputs['cpu'] = dy


class MeanElement(UserGraph):

    name = 'Mean'

    def __init__(self, previous_elements=None, axis=None, keepdims=False):
        fwd_op = mean_forward(axis=axis, keepdims=keepdims) if rm.is_cuda_active(
        ) else mean_forward_cpu(axis=axis, keepdims=keepdims)
        bwd_ops = [mean_backward(fwd_op) if rm.is_cuda_active() else mean_backward_cpu(fwd_op)]
        super().__init__(fwd_op, bwd_ops, previous_elements)


@populate_graph
class Mean(GraphFactory):

    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def connect(self, other):
        ret = MeanElement([other], axis=self.axis, keepdims=self.keepdims)
        return ret


@populate_graph
@populate_basics
def mean(self, axis=None, keepdims=False):
    return MeanElement([self], axis=axis, keepdims=keepdims)


UserGraph.mean = mean
