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


class concatenate_forward(operation):
    '''Concatenate forward operation class.
    '''

    name = 'Concatenate (F)'

    def __init__(self, axis=None):
        self.axis = axis

    def setup(self, inputs):
        '''Prepares workspaces for this operation.

        Args:
            inputs (list of GraphMultiStorage): Input data to this operation.

        concatenate_forward class requires inputs to contain following keys.

        +-------+-----+------------------------------------+
        | Index | Key |              Role                  |
        +=======+=====+====================================+
        |   n   |  y  | Output of nth previous operation.  |
        +-------+-----+------------------------------------+
        '''

        assert isinstance(inputs, (list, tuple)), \
            "Concatenate accepts only list or tuple of array."
        inputs = [a['y'] for a in inputs]
        self._inputs = inputs
        gpus = inputs[0].gpus
        axis = self.axis

        out_shape = inputs[0].shape[:axis] + \
            (int(np.sum([int(a.shape[axis]) for a in inputs])), ) + inputs[0].shape[axis + 1:]

        self.gpus = gpus
        outs = GraphMultiStorage(shape=out_shape, gpus=gpus)
        self._split_index = np.cumsum([a.shape[axis] for a in inputs[:-1]]).tolist()
        self._outputs = outs
        self._vars = {'y': outs}

    def perform(self):
        if self.axis == 0:
            val = np.sum(a.shape[0].value for a in self._inputs)
            self._outputs.shape[0].value = val
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cuconcat([a[gpu] for a in self._inputs], self._outputs[gpu], axis=self.axis)


class concatenate_forward_cpu(concatenate_forward):

    def perform(self):
        ret = np.concatenate([a['cpu'] for a in self._inputs], axis=self.axis)
        self._outputs['cpu'] = ret


class concatenate_backward(operation):
    '''Concatenate backward operation class.
    '''

    name = 'Concatenate (B)'

    def __init__(self, associated_forward, nth_input):
        self._fwd_op = associated_forward
        self._nth_input = nth_input

    def setup(self, inputs):
        '''Prepares workspaces for this operation.

        Args:
            inputs (list of GraphMultiStorage): Input data to this operation.

        concatenate_backward class requires inputs to contain following keys.

        +-------+-----+------------------------------------+
        | Index | Key |              Role                  |
        +=======+=====+====================================+
        |   0   |  y  | Output of previous operation.      |
        +-------+-----+------------------------------------+
        '''

        n = self._nth_input
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        fwd_inputs = self._fwd_op._inputs[n]
        out_shape = fwd_inputs.shape
        outs = GraphMultiStorage(shape=out_shape, gpus=gpus)

        self.gpus = gpus
        self._inputs = inputs
        self._outputs = outs
        self._split_index = self._fwd_op._split_index
        self.axis = self._fwd_op.axis
        self._vars = {'y': outs, 'dy': outs, id(fwd_inputs): outs}

    def perform(self):
        n = self._nth_input
        axis = self.axis
        sp_index = self._split_index
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            splitted = self._inputs[gpu].split(sp_index, axis=axis)
            self._outputs[gpu] = splitted[n]


class concatenate_backward_cpu(concatenate_backward):

    def perform(self):
        n = self._nth_input
        axis = self.axis
        sp_index = self._split_index
        splitted = np.split(self._inputs['cpu'], sp_index, axis=axis)
        self._outputs['cpu'] = splitted[n]


class ConcatenateElement(UserGraph):

    _name = 'Concatenate'

    def __init__(self, previous_elements=None, axis=0):
        fwd_op = concatenate_forward(axis=axis) if rm.is_cuda_active() \
            else concatenate_forward_cpu(axis=axis)
        bwd_ops = [concatenate_backward(fwd_op, nth) if rm.is_cuda_active()
                   else concatenate_backward_cpu(fwd_op, nth) for nth in range(len(previous_elements))]
        super().__init__(fwd_op, bwd_ops, previous_elements)


@populate_graph
class Concatenate(GraphFactory):
    '''A factory class of concatenate function element.

    Example:
        >>> import numpy as np
        >>> import renom.graph as rmg
        >>>
        >>> m1 = np.arange(4).reshape(2, 2)
        >>> m2 = np.arange(2).reshape(2, 1)
        >>> print(rmg.concatenate([m1, m2], axis=1))
        Concatenate (F):
        [[0. 1. 0.]
         [2. 3. 1.]]

    '''

    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def connect(self, other):
        assert isinstance(other, (list, tuple)), \
            "Concatenate accepts only list or tuple of array."
        return ConcatenateElement(other, axis=self.axis)


@populate_graph
@populate_basics
def concatenate(elems, axis=0):
    '''A function style factory of concatenate function element.

    Args:
        elems (list, tuple): Array to concatenated.
        axis (int): Concatenation will be performed along this axis.

    For more information, please refer :py:class:`~renom.graph.basics.concatenate_element.Concatenate`.
    '''

    assert isinstance(elems, (list, tuple)), \
        "Concatenate accepts only list or tuple of array."
    return Concatenate(axis=axis)(elems)
