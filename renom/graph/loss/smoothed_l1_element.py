#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

import numpy as np

import renom as rm
from renom.graph.core import UserLossGraph, operation, GraphMultiStorage, GraphFactory
from renom.graph import populate_graph


class smooth_l1_forward(operation):

    name = 'Smooth L1 (F)'
    roles = ['loss']

    def __init__(self, delta=1.0):
        self._delta = delta

    def setup(self, inputs):
        predictions = inputs[0]['y']
        real_values = inputs[1]['y']
        self.gpus = predictions.gpus
        self._graph_input = predictions
        self._label_input = real_values

        out_shape = (1, )
        assert predictions.shape == real_values.shape
        output = GraphMultiStorage(shape=out_shape, gpus=predictions.gpus)

        self._vars = {'y': output}
        self._outputs = output
        self._N = predictions.shape[0]

    def perform(self):
        self._d = GraphMultiStorage(shape=self._graph_input.shape, gpus=self.gpus)
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            x = self._graph_input[gpu].new_array()
            y = self._label_input[gpu].new_array()
            N = len(x)
            d = x - y
            delta = self._delta
            abs_d = abs(d)
            flag = abs_d < delta
            ret = np.sum(flag * 0.5 * (d * d) +
                         (1 - flag) * (abs_d - 0.5 * delta) * delta)
            ret = ret.reshape(1,) / N
            self._d[gpu] = d
            self._outputs[gpu].to_gpu(ret)


class smooth_l1_forward_cpu(smooth_l1_forward):

    def perform(self):
        x = self._graph_input['cpu']
        y = self._label_input['cpu']
        N = len(x)
        d = x - y
        delta = self._delta
        abs_d = abs(d)
        flag = abs_d < delta
        ret = np.sum(flag * 0.5 * (d * d) +
                     (1 - flag) * (abs_d - 0.5 * delta) * delta)
        ret = ret.reshape(1,) / N
        self._d = d
        self._outputs['cpu'] = ret


class smooth_l1_backward(operation):

    name = 'Smooth L1 (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward
        self._delta = self._fwd_op._delta

    def setup(self, inputs):

        if len(inputs) > 3:
            self._dy = inputs[3]['y']
        else:
            self._dy = None
        predictions = inputs[0]['y']
        real_values = inputs[1]['y']
        self._graph_input = predictions
        self._label_input = real_values
        gpus = predictions.gpus
        self.gpus = gpus
        output = GraphMultiStorage(shape=predictions.shape, gpus=gpus)
        self._outputs = output
        self._vars = {'y': output, 'dy': output, id(self._fwd_op._graph_input): output}
        self._N = predictions.shape[0]

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            if self._dy is not None:
                dy = self._dy[gpu]
            else:
                dy = 1
            d = self._fwd_op._d[gpu]
            N = len(d)
            delta = self._delta
            mask = abs(d) <= delta
            sign = (d > 0) * 2 - 1
            dx = mask * d + (1 - mask) * sign * delta
            ret = dx / N
            self._outputs[gpu].to_gpu(ret)
            rm.cuda.cumul(self._outputs[gpu], dy, self._outputs[gpu], handle)


class smooth_l1_backward_cpu(smooth_l1_backward):

    def perform(self):
        if self._dy is not None:
            dy = self._dy['cpu']
        else:
            dy = 1
        d = self._fwd_op._d
        N = len(d)
        delta = self._delta
        mask = abs(d) <= delta
        sign = (d > 0) * 2 - 1
        dx = mask * d + (1 - mask) * sign * delta
        ret = dx * dy / N
        self._outputs['cpu'] = ret


class SmoothL1Element(UserLossGraph):

    def __init__(self, delta=1.0, previous_elements=None):
        self._delta = delta
        fwd_op = smooth_l1_forward(
            delta) if rm.is_cuda_active() else smooth_l1_forward_cpu(delta)
        bwd_ops = [smooth_l1_backward(fwd_op) if rm.is_cuda_active()
                   else smooth_l1_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


@populate_graph
class SmoothL1(GraphFactory):
    """A factory class of smooth l1 loss function element.

    Args:
        delta (float):


    """

    def prepare(self, delta=1.0):
        self._delta = delta

    def connect(self, predictions, true_values):
        ret = SmoothL1Element(self._delta, previous_elements=[predictions, true_values])
        return ret
