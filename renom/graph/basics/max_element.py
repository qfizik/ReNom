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


class max_forward(operation):
    '''Max forward operation class.
    '''

    name = 'Max (F)'

    def __init__(self, axis=None, keepdims=True):
        assert isinstance(axis, (int, type(None))), "Only accepts axis as integer or None."
        self.axis = axis
        self.keepdims = keepdims

    def setup(self, inputs):
        '''Prepares workspaces for this operation.

        Args:
            inputs (list of GraphMultiStorage): Input data to this operation.

        max_forward class requires inputs to contain following keys.

        +-------+-----+------------------------------------+
        | Index | Key |              Role                  |
        +=======+=====+====================================+
        |   0   |  y  | Output of 1st previous operation.  |
        +-------+-----+------------------------------------+
        '''

        inputs = inputs[0]['y']
        self._inputs = inputs
        gpus = inputs.gpus
        self.gpus = gpus
        if self.axis is None and not self.keepdims:
            out_shape = (1, )
        else:
            out_shape = np.max(np.zeros(inputs.shape, dtype=np.bool),
                               axis=self.axis, keepdims=self.keepdims).shape
            if not out_shape:
                out_shape = (1, )
        outs = GraphMultiStorage(shape=out_shape, gpus=gpus)
        self._outputs = outs
        self._vars = {'y': outs}
        self._indexes = {}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            r = rm.cuda.cu_reduce_max(
                self._inputs[gpu], handle, axis=self.axis, keepdims=self.keepdims)
            # This is required for backward.
            self._indexes[gpu] = rm.cuda.cu_reduce_argmax(self._inputs[gpu], handle, axis=self.axis)
            self._outputs[gpu].copy_from(r)


class max_forward_cpu(max_forward):

    def perform(self):
        ret = np.max(self._inputs['cpu'], axis=self.axis, keepdims=self.keepdims)
        self._indexes['cpu'] = np.argmax(self._inputs['cpu'], axis=self.axis)
        self._outputs['cpu'] = ret


class max_backward(operation):
    '''Max backward operation class.
    '''

    name = 'Max (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        '''Prepares workspaces for this operation.

        Args:
            inputs (list of GraphMultiStorage): Input data to this operation.

        max_backward class requires inputs to contain following keys.

        +-------+-----+------------------------------------+
        | Index | Key |              Role                  |
        +=======+=====+====================================+
        |   0   |  y  | Output of previous operation.      |
        +-------+-----+------------------------------------+
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
        self.indexes = self._fwd_op._indexes
        self._zeros = GraphMultiStorage(
            shape=fwd_inputs.shape, gpus=gpus, initializer=init.Constant(0))
        self._vars = {'y': outs, id(fwd_inputs): outs}

    def perform(self):
        axis = self.axis
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            dy = self._inputs[gpu]
            zeros = self._zeros[gpu]
            index = self.indexes[gpu].new_array().astype(np.int)
            if axis is None:
                # Copy data to the pointed adress.
                zeros_prt = zeros.reshape(-1)
                zeros_prt[int(index)] = dy
                zeros = zeros_prt.reshape(zeros.shape)
            else:
                axis_list = list(range(len(zeros.shape)))
                axis_list.pop(axis)
                axis_list.append(axis)
                rev = [-1] * len(axis_list)
                for i, a in enumerate(axis_list):
                    rev[a] = i
                zeros_prt = zeros.transpose(axis_list)
                if(not self.keepdims):
                    pass
                else:
                    axis_list = list(range(len(dy.shape)))
                    axis_list.pop(axis)
                    axis_list.append(axis)
                    rev = [-1] * len(axis_list)
                    for i, a in enumerate(axis_list):
                        rev[a] = i
                    dy = dy.transpose(axis_list)
                for i in np.ndindex(index.shape):
                    temp = zeros_prt[i]
                    temp[int(index[i])] = dy[i]
                    zeros_prt[i] = temp

                # Reverse Transpose
                zeros = zeros_prt.transpose(np.argsort(axis_list).tolist())
            self._outputs[gpu] = zeros


class max_backward_cpu(max_backward):

    def perform(self):
        axis = self.axis
        dy = self._inputs['cpu']
        zeros = self._zeros['cpu']
        index = self.indexes['cpu']

        if axis is None:
            # Copy data to the pointed adress.
            zeros_prt = zeros.reshape(-1)
            zeros_prt[index] = dy
        else:
            axis_list = list(range(len(zeros.shape)))
            axis_list.pop(axis)
            axis_list.append(axis)
            rev = [-1] * len(axis_list)
            for i, a in enumerate(axis_list):
                rev[a] = i
            zeros_prt = np.transpose(zeros, axis_list)
            if(not self.keepdims):
                pass
            else:
                axis_list = list(range(len(dy.shape)))
                axis_list.pop(axis)
                axis_list.append(axis)
                rev = [-1] * len(axis_list)
                for i, a in enumerate(axis_list):
                    rev[a] = i
                dy = np.transpose(dy, axis_list)
            for i in np.ndindex(index.shape):
                zeros_prt[i][index[i]] = dy[i]
        self._outputs['cpu'] = zeros


class MaxElement(UserGraph):

    name = 'Max'

    def __init__(self, previous_elements=None, axis=None, keepdims=False):
        fwd_op = max_forward(axis=axis, keepdims=keepdims) if rm.is_cuda_active(
        ) else max_forward_cpu(axis=axis, keepdims=keepdims)
        bwd_ops = [max_backward(fwd_op) if rm.is_cuda_active() else max_backward_cpu(fwd_op)]
        super().__init__(fwd_op, bwd_ops, previous_elements)


@populate_graph
class Max(GraphFactory):
    '''A factory class of max function element.
    Max operation of the UserGraph object will call this factory class.

    Example:
        >>> import numpy as np
        >>> import renom.graph as rmg
        >>>
        >>> x = np.arange(1, 7).reshape(2, 3)
        >>> layer = rmg.Max()
        >>> print(layer(x1))
        Max (F):
        6.0

    '''


    def connect(self, other):
        ret = MaxElement(previous_elements=[other])
        return ret


@populate_graph
@populate_basics
def max(self, axis=None, keepdims=False):
    '''A function style factory of max operation element.

    Args:
        self (UserGraph): Left hand input.
        axis (int, None): Max operation will be performed along this axis.
        keepdims (bool): If Ture is given, the original axis will be remained as 1.

    For more information, please refer :py:class:`~renom.graph.basics.max_element.Max`.
    '''

    return MaxElement([self], axis=axis, keepdims=keepdims)


UserGraph.max = max
