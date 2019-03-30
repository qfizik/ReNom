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


class sum_forward(operation):
    '''Sum forward operation class.

    Args:
        axis (int, tuple, None): Summation will be performed along given axis.
        keepdims (bool): If Ture is given, the original axis will be remained as 1.

    '''

    name = 'Sum (F)'

    def __init__(self, axis=None, keepdims=True):
        self.axis = axis
        self.keepdims = keepdims

    def setup(self, inputs):
        '''Prepares workspaces for this operation.

        Args:
            inputs (list of GraphMultiStorage): Input data to this operation.

        sum_forward class requires inputs to contain following keys.

        +-------+-----+--------------------------------+
        | Index | Key |              Role              |
        +=======+=====+================================+
        |   0   |  y  | Output of previous operation.  |
        +-------+-----+--------------------------------+
        '''

        inputs = inputs[0]['y']
        self._inputs = inputs
        gpus = inputs.gpus
        self.gpus = gpus
        if self.axis is None and not self.keepdims:
            out_shape = (1, )
        else:
            out_shape = np.sum(np.zeros(inputs.shape, dtype=np.bool),
                               axis=self.axis, keepdims=self.keepdims).shape
            if not out_shape:
                out_shape = (1, )
        outs = GraphMultiStorage(shape=out_shape, gpus=gpus)
        self._outputs = outs
        self._vars = {'y': outs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            r = rm.cuda.cusum(self._inputs[gpu], handle, axis=self.axis, keepdims=self.keepdims)
            self._outputs[gpu].copy_from(r)


class sum_forward_cpu(sum_forward):

    def perform(self):
        ret = np.sum(self._inputs['cpu'], axis=self.axis, keepdims=self.keepdims)
        self._outputs['cpu'] = ret


class sum_backward(operation):
    '''Sum backward operation class.
    '''

    name = 'Sum (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        '''Prepares workspaces for this operation.

        Args:
            inputs (list of GraphMultiStorage): Input data to this operation.

        sum_backward class requires inputs to contain following keys.

        +-------+-----+--------------------------------+
        | Index | Key |              Role              |
        +=======+=====+================================+
        |   0   |  y  | Output of previous operation.  |
        +-------+-----+--------------------------------+
        '''

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
        self._ones = GraphMultiStorage(
            shape=fwd_inputs.shape, gpus=gpus, initializer=init.Constant(1))
        self._vars = {'y': outs, 'dy': outs, id(fwd_inputs): outs}

    def perform(self):
        axis = self.axis
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            dy = self._inputs[gpu].reshape(self.expand_shape)
            ones = self._ones[gpu]
            self._outputs[gpu] = dy * ones
            if axis is None:
                self._outputs[gpu] = ones * dy
            else:
                if not self.keepdims:
                    dy = ones * dy.reshape(self.expand_shape)
                else:
                    dy = ones * dy
                self._outputs[gpu] = dy


class sum_backward_cpu(sum_backward):

    def perform(self):
        axis = self.axis
        dy = self._inputs['cpu'].reshape(self.expand_shape)
        ones = self._ones['cpu']
        if axis is None:
            self._outputs['cpu'] = ones * dy
        else:
            if not self.keepdims:
                dy = ones * dy.reshape(self.expand_shape)
            else:
                dy = ones * dy
            self._outputs['cpu'] = dy


class SumElement(UserGraph):

    name = 'Sum'

    def __init__(self, previous_elements=None, axis=None, keepdims=False):
        fwd_op = sum_forward(axis=axis, keepdims=keepdims) if rm.is_cuda_active(
        ) else sum_forward_cpu(axis=axis, keepdims=keepdims)
        bwd_ops = [sum_backward(fwd_op) if rm.is_cuda_active() else sum_backward_cpu(fwd_op)]
        super().__init__(fwd_op, bwd_ops, previous_elements)


@populate_graph
class Sum(GraphFactory):
    '''A factory class of sum function element.

    Args:
        axis (int, tuple, None): Summation will be performed along given axis.
        keepdims (bool): If Ture is given, the original axis will be remained as 1.


    Example:
        >>> import numpy as np
        >>> import renom.graph as rmg
        >>>
        >>> x = np.arange(6).reshape(2, 3)
        >>> rmg.sum(x)
        Sum (F):
        15.0
        >>> rmg.sum(x, axis=1)
        Sum (F):
        [ 3. 12.]
        >>> rmg.sum(x, axis=0) 
        Sum (F):
        [3. 5. 7.]

    '''

    def prepare(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def connect(self, other):
        ret = SumElement(previous_elements=[other], axis=self.axis, keepdims=self.keepdims)
        return ret


@populate_graph
@populate_basics
def sum(self, axis=None, keepdims=False):
    '''A function style factory of sum function element.

    Args:
        self (UserGraph, ndarray): Input array.
        axis (int, tuple, None): Summation will be performed along given axis.
        keepdims (bool): If Ture is given, the original axis will be remained as 1.

    For more information, please refer :py:class:`~renom.graph.basics.sum_element.Sum`.
    '''
    return Sum(axis=axis, keepdims=keepdims)(self)


UserGraph.sum = sum
