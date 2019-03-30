#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

import numpy as np

from renom.graph.core import operation, operational_element, \
    UserGraph, GraphMultiStorage, GraphFactory
import renom as rm
from renom.graph import populate_graph


class reshape_forward(operation):
    '''Reshape forward operation class.
    '''

    name = 'Reshape (F)'

    def __init__(self, shape):
        self._new_shape = shape

    def setup(self, inputs):
        '''Prepares workspaces for this operation.

        Args:
            inputs (list of GraphMultiStorage): Input data to this operation.

        reshape_forward class requires inputs to contain following keys.

        +-------+-----+------------------------------------+
        | Index | Key |              Role                  |
        +=======+=====+====================================+
        |   0   |  y  | Output of 1st previous operation.  |
        +-------+-----+------------------------------------+
        '''

        self._inputs = inputs[0]['y']
        new_shape = [self._inputs.shape[0]]
        new_shape.extend(self._new_shape)
        new_shape = np.empty(self._inputs.shape).reshape(new_shape).shape
        gpus = self._inputs.gpus
        self._outputs = GraphMultiStorage(shape=new_shape, gpus=gpus, ptrs=self._inputs)
        self._vars = {'y': self._outputs}
        print(self._inputs)

    def perform(self):
        pass


class reshape_backward(operation):
    '''Reshape backward operation class.
    '''

    name = 'Reshape (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        '''Prepares workspaces for this operation.

        Args:
            inputs (list of GraphMultiStorage): Input data to this operation.

        reshape_backward class requires inputs to contain following keys.

        +-------+-----+------------------------------------+
        | Index | Key |              Role                  |
        +=======+=====+====================================+
        |   0   |  y  | Output of previous operation.      |
        +-------+-----+------------------------------------+
        '''

        self._inputs = inputs[0]['y']
        shape = self._fwd_op._inputs.shape
        gpus = self._inputs.gpus
        fwd_op_inputs = self._fwd_op._inputs
        self._outputs = GraphMultiStorage(shape=shape, gpus=gpus, ptrs=self._inputs)
        self._vars = {'y': self._outputs, 'dy': self._outputs, id(fwd_op_inputs): fwd_op_inputs}

    def perform(self):
        pass

    def __repr__(self):
        return self._outputs.__repr__()


class ReshapeElement(UserGraph):

    def __init__(self, shape, previous_element=None):
        self._shape = shape
        fwd_op = reshape_forward(shape)
        bwd_ops = [reshape_backward(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_element)


@populate_graph
class Reshape(GraphFactory):
    '''A factory class of reshape function element.
    Reshape operation of the UserGraph object will call this factory class.

    Example:





    '''


    def __init__(self, shape):
        super().__init__()
        self.shp = shape

    def connect(self, other):
        ret = ReshapeElement(self.shp, other)
        return ret


@populate_graph
def reshape(self, shape):
    '''A function style factory of reshape operation element.

    Args:
        self (UserGraph, ndarray): Input array.
        shape (list, int, None): Indices.

    For more information, please refer :py:class:`~renom.graph.basics.reshape_element.ReshapeElement`.
    '''

    return Reshape(shape)(self)

UserGraph.reshape = reshape
