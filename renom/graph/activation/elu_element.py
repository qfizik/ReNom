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


class elu_forward(operation):
    '''Elu forward operation class.

    Args:
        alpha (float): Coefficient used in elu.
    '''

    name = 'Elu (F)'

    def __init__(self, alpha):
        self._alpha = alpha

    def setup(self, inputs):
        '''Prepares workspaces for this operation.

        Args:
            inputs (list of GraphMultiStorage): Input data to this operation.

        elu_forward class requires inputs to contain following keys.

        +-------+-----+--------------------------------+
        | Index | Key |              Role              |
        +=======+=====+================================+
        |   0   |  y  | Output of previous operation.  |
        +-------+-----+--------------------------------+
        '''

        x = inputs[0]['y']
        gpus = x.gpus
        self.gpus = gpus
        input_shape = x.shape
        outs = GraphMultiStorage(shape=input_shape, gpus=gpus)
        self._inputs = x
        self._outputs = outs
        self._vars = {'y': outs}

    def perform(self):
        '''This function performs operation with gpu.
        '''
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cueru_forward(self._alpha, self._inputs[gpu], self._outputs[gpu])


class elu_forward_cpu(elu_forward):

    def perform(self):
        x = self._inputs['cpu']
        ret = np.where(x > 0, x, (np.exp(x) - 1) * self._alpha)
        self._outputs['cpu'] = ret


class elu_backward(operation):
    '''Elu backward operation class.

    Args:
        associated_forward (forward_operation): Corresponding forward operation.
    '''

    name = 'Elu (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward
        self._alpha = self._fwd_op._alpha

    def setup(self, inputs):
        '''Prepares workspaces for this operation.

        Args:
            inputs (list of GraphMultiStorage): Input data to this operation.

        elu_forward class requires inputs to contain following keys.

        +-------+-----+--------------------------------+
        | Index | Key |              Role              |
        +=======+=====+================================+
        |   0   |  y  | Output of previous operation.  |
        +-------+-----+--------------------------------+
        '''

        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus
        in_shape = inputs.shape
        outs = GraphMultiStorage(shape=in_shape, gpus=gpus)
        self._inputs = inputs
        self._outputs = outs
        self._fwd_in = self._fwd_op._inputs
        self._vars = {'y': outs, id(self._fwd_op._inputs): outs}

    def perform(self):
        '''This function performs operation with cpu.
        '''
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cueru_backward(self._alpha, self._fwd_in[gpu], self._outputs[gpu])
            rm.cu.cumul(self._outputs[gpu], self._inputs[gpu], self._outputs[gpu], handle)


class elu_backward_cpu(elu_backward):

    def perform(self):
        dy = self._inputs['cpu']
        y = self._fwd_op._outputs['cpu']
        alpha = self._alpha
        ret = np.where(y > 0, dy, (alpha + y) * dy)
        self._outputs['cpu'] = ret


class EluElement(UserGraph):
    '''Expansion of UserGraph class to Elu.

    Args:
        alpha (float): Coefficient used in elu.
        previous_elements (None, list): List of previous user graph elements.

    '''

    def __init__(self, alpha=0.01, previous_elements=None):
        fwd_op = elu_forward(alpha) if rm.is_cuda_active() else elu_forward_cpu(alpha)
        bwd_ops = [elu_backward(fwd_op) if rm.is_cuda_active() else elu_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


class Elu(GraphFactory):
    '''A factory class of elu activation function element. [elu]_

    .. math::

        y = max(x, \\alpha * exp(x) - 1)

    Args:
        alpha (float): Alpha coefficient for Elu.

    Example:
        >>> import numpy as np
        >>> import renom.graph as rmg
        >>>
        >>> x = np.array([-1, 0, 1])
        >>>
        >>> layer = rmg.Elu()
        >>> layer(x)
        Elu (F):
        [-0.00632121  0.          1.        ]
        >>>
        >>> # Create element using function interface.
        >>> rmg.elu(x)
        Elu (F):
        [-0.00632121  0.          1.        ]


    .. [elu] Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter.
        Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs).
        ICLR, 2016.

    '''

    def prepare(self, alpha=0.01):
        self._alpha = alpha

    def connect(self, other):
        ret = EluElement(alpha=self._alpha, previous_elements=other)
        return ret

def elu(x, alpha=0.01):
    '''A function style factory of elu activation function element.

    Args:
        alpha (float): Coefficient used in elu.

    For more information, please refer :py:class:`~renom.graph.activation.elu_element.Elu`.
    '''
    return EluElement(alpha=alpha, previous_elements=[x])
