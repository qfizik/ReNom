#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

import numpy as np

import renom as rm
from renom.graph.core import operation, operational_element, \
    UserGraph, GraphMultiStorage, GraphFactory
from renom.graph import populate_graph
from renom.graph.basics import populate_basics


class square_forward(operation):
    '''Square forward operation class.
    '''

    name = 'Square (F)'

    def setup(self, inputs):
        '''Prepares workspaces for this operation.

        Args:
            inputs (list of GraphMultiStorage): Input data to this operation.

        square_forward class requires inputs to contain following keys.

        +-------+-----+------------------------------------+
        | Index | Key |              Role                  |
        +=======+=====+====================================+
        |   0   |  y  | Output of 1st previous operation.  |
        +-------+-----+------------------------------------+
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
            self._outputs[gpu] = self._inputs[gpu] * self._inputs[gpu]


class square_forward_cpu(square_forward):

    def perform(self):
        self._outputs['cpu'] = self._inputs['cpu'] * self._inputs['cpu']


class square_backward(operation):
    '''Square backward operation class.
    '''

    name = 'Square (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        inputs = inputs[0]['y']
        fwd_inputs = self._fwd_op._inputs
        shape = fwd_inputs.shape
        gpus = fwd_inputs.gpus

        self.gpus = gpus
        self._inputs = inputs
        self._fwd_inputs = fwd_inputs
        self._outputs = GraphMultiStorage(shape=shape, gpus=gpus)
        self._vars = {'y': self._outputs, 'dy': self._outputs, id(fwd_inputs): self._outputs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            self._outputs[gpu] = self._fwd_inputs[gpu] * 2 * self._inputs[gpu]


class square_backward_cpu(square_backward):

    def perform(self):
        self._outputs['cpu'] = self._fwd_inputs['cpu'] * 2 * self._inputs['cpu']


class SquareElement(UserGraph):

    _name = 'Square Element'

    def __init__(self, previous_element=None):
        fwd_op = square_forward() if rm.is_cuda_active() else square_forward_cpu()
        bwd_ops = [square_backward(fwd_op) if rm.is_cuda_active() else square_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops,
                         previous_elements=previous_element)

@populate_graph
@populate_basics
class Square(GraphFactory):

    def connect(self, x):
        return SquareElement(previous_element=[x])


@populate_graph
@populate_basics
def square(self):
    ret = Square()(self)
    return ret


UserGraph.square = square
