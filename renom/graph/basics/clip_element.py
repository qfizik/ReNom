#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

import numpy as np

from renom.graph.core import operation, UserGraph, GraphMultiStorage, GraphFactory
import renom as rm
from renom.graph import populate_graph

class clip_forward(operation):

    name = 'Clip (F)'

    def __init__(self, floor, ceil, use_key=None):
        self.floor = floor
        self.ceil = ceil
        self.key = use_key

    def setup(self, inputs):
        key = self.key
        if key is None:
            key = 'y'
        inputs = inputs[0][key]
        output_shape = inputs.shape
        gpus = inputs.gpus

        self._inputs = inputs
        self.gpus = gpus
        self._outputs = GraphMultiStorage(shape=output_shape, gpus=gpus)
        self._vars = {'y': self._outputs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cu_clip(self._inputs[gpu], self.floor, self.ceil, self._outputs[gpu])


class clip_forward_cpu(clip_forward):

    def perform(self):
        self._outputs['cpu'] = np.clip(self._inputs['cpu'], self.floor, self.ceil)


class clip_backward(operation):

    name = 'Clip (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward
        self.floor = self._fwd_op.floor
        self.ceil = self._fwd_op.ceil

    def setup(self, inputs):
        inputs = inputs[0]['y']
        fwd_inputs = self._fwd_op._inputs
        shape = fwd_inputs.shape
        gpus = fwd_inputs.gpus

        self.gpus = gpus
        self._inputs = inputs
        self._fwd_ins = self._fwd_op._inputs
        self._outputs = GraphMultiStorage(shape=shape, gpus=gpus)
        self._vars = {'y': self._outputs, 'dy': self._outputs, id(fwd_inputs): self._outputs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cu_clip_back(self._fwd_ins[gpu], self.floor, self.ceil, self._outputs[gpu])
            rm.cuda.cumul(self._inputs[gpu], self._outputs[gpu], self._outputs[gpu], handle)


class clip_backward_cpu(clip_backward):

    def perform(self):
        x = self._fwd_ins['cpu']
        y = np.ones_like(x)
        y[x < self.floor] = 0
        y[x > self.ceil] = 0
        y *= self._inputs['cpu']
        self._outputs['cpu'] = y


@populate_graph
class ClipElement(UserGraph):

    _name = 'Clip Element'

    def __init__(self, floor, ceil, previous_element=None):
        fwd_op = clip_forward(floor, ceil) if rm.is_cuda_active() else clip_forward_cpu(floor, ceil)
        bwd_ops = [clip_backward(fwd_op) if rm.is_cuda_active() else clip_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops,
                         previous_elements=previous_element)


def clip(self):
    ret = ClipElement([self])
    return ret


UserGraph.clip = clip
