#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

import numpy as np
import renom as rm
from renom.graph.core import UserGraph, operation, GraphFactory, \
    graph_variable, GraphMultiStorage


class sigmoid_forward(operation):
    '''Sigmoid forward operation class.
    '''

    name = 'Sigmoid (F)'

    def setup(self, inputs):
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus
        outs = GraphMultiStorage(shape=inputs.shape, gpus=gpus)
        self._vars = {'y': outs}
        self._inputs = inputs
        self._outputs = outs

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cusigmoid(self._inputs[gpu], self._outputs[gpu])


class sigmoid_forward_cpu(sigmoid_forward):

    def perform(self):
        x = self._inputs['cpu']
        ret = 1. / (1. + np.exp(-x))
        self._outputs['cpu'] = ret


class sigmoid_backward(operation):

    name = 'Sigmoid (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        '''Prepares workspaces for this operation.

        Args:
            inputs (list of GraphMultiStorage): Input data to this operation.

        sigmoid_forward class requires inputs to contain following keys.

        +-------+-----+--------------------------------+
        | Index | Key |              Role              |
        +=======+=====+================================+
        |   0   |  y  | Output of previous operation.  |
        +-------+-----+--------------------------------+
        '''

        inputs = inputs[0]['dy']
        gpus = inputs.gpus
        self.gpus = gpus
        outs = GraphMultiStorage(shape=inputs.shape, gpus=gpus)
        self._vars = {'y': outs, 'dy': outs, id(self._fwd_op._inputs): outs}
        self._fwd_out = self._fwd_op._outputs
        self._inputs = inputs
        self._outputs = outs

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cumul(self._fwd_out[gpu], -1, self._outputs[gpu], handle)
            rm.cuda.cuadd(self._outputs[gpu], 1, self._outputs[gpu], handle)
            rm.cuda.cumul(self._fwd_out[gpu], self._outputs[gpu], self._outputs[gpu], handle)
            rm.cuda.cumul(self._inputs[gpu], self._outputs[gpu], self._outputs[gpu], handle)


class sigmoid_backward_cpu(sigmoid_backward):

    def perform(self):
        dy = self._inputs['cpu']
        y = self._fwd_out['cpu']
        ret = y * (1. - y) * dy
        self._outputs['cpu'] = ret


class SigmoidElement(UserGraph):

    def __init__(self, previous_elements=None):
        fwd_op = sigmoid_forward() if rm.is_cuda_active() else sigmoid_forward_cpu()
        bwd_ops = [sigmoid_backward(fwd_op) if rm.is_cuda_active()
                   else sigmoid_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


class Sigmoid(GraphFactory):
    '''A factory class of sigmoid activation function element.

    .. math::

        y = 1/(exp(-x) + 1)

    Example:
        >>> import numpy as np
        >>> import renom.graph as rmg
        >>>
        >>> x = np.array([-1, 0, 1])
        >>>
        >>> layer = rmg.Sigmoid()
        >>> layer(x)
        Sigmoid (F):
        [0.26894143 0.5        0.7310586 ]
        >>>
        >>> # Create element using function interface.
        >>> rmg.sigmoid(x)
        Sigmoid (F):
        [0.26894143 0.5        0.7310586 ]

    '''

    def connect(self, other):
        ret = SigmoidElement(previous_elements=other)
        return ret


def sigmoid(x):
    return SigmoidElement(previous_elements=[x])
