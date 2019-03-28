#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

import numpy as np

import renom as rm
from renom.graph.core import operation, UserGraph, GraphMultiStorage, GraphFactory
from renom.graph import populate_graph


class clip_forward(operation):
    '''Clip forward operation class.

    Args:
        floor (float): The lower bound of clipping.
        ceil (float): The upper bound of clipping.

    '''

    name = 'Clip (F)'

    def __init__(self, floor, ceil):
        self.floor = floor
        self.ceil = ceil

    def setup(self, inputs):
        '''Prepares workspaces for this operation.

        Args:
            inputs (list of GraphMultiStorage): Input data to this operation.

        clip_forward class requires inputs to contain following keys.

        +-------+-----+--------------------------------+
        | Index | Key |              Role              |
        +=======+=====+================================+
        |   0   |  y  | Output of previous operation.  |
        +-------+-----+--------------------------------+
        '''

        inputs = inputs[0]['y']
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
    '''Clip backward operation class.
    '''

    name = 'Clip (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward
        self.floor = self._fwd_op.floor
        self.ceil = self._fwd_op.ceil

    def setup(self, inputs):
        '''Prepares workspaces for this operation.

        Args:
            inputs (list of GraphMultiStorage): Input data to this operation.

        clip_backward class requires inputs to contain following keys.

        +-------+-----+--------------------------------+
        | Index | Key |              Role              |
        +=======+=====+================================+
        |   0   |  y  | Output of previous operation.  |
        +-------+-----+--------------------------------+
        '''

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

    def __init__(self, floor, ceil, previous_elements=None):
        fwd_op = clip_forward(floor, ceil) if rm.is_cuda_active() else clip_forward_cpu(floor, ceil)
        bwd_ops = [clip_backward(fwd_op) if rm.is_cuda_active() else clip_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops,
                         previous_elements=previous_elements)

@populate_graph
class Clip(GraphFactory):
    '''A factory class of clip function element.

    Args:
        floor (float): The lower bound of clipping.
        ceil (float): The upper bound of clipping.

    Example:
        >>> import numpy as np
        >>> import renom.graph as rmg
        >>> 
        >>> x = np.arange(6).reshape(2, 3)
        >>> 
        >>> rmg.clip(x, 0, 1)
        Clip (F):
        [[0. 1. 1.]
         [1. 1. 1.]]

    '''

    def prepare(self, floor, ceil):
        self.floor = floor
        self.ceil = ceil

    def connect(self, other):
        return ClipElement(self.floor, self.ceil, previous_elements=[other])


@populate_graph
def clip(self, floor, ceil):
    '''A function style factory of clip function element.

    Args:
        floor (float): The lower bound of clipping.
        ceil (float): The upper bound of clipping.

    For more information, please refer :py:class:`~renom.graph.basics.clip_element.Clip`.
    '''
    ret = Clip(floor, ceil)(self)
    return ret


UserGraph.clip = clip
