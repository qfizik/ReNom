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
from renom.graph.basics import populate_basics


class log_forward(operation):
    '''Log forward operation class.
    '''

    name = 'Log (F)'

    def setup(self, inputs):
        '''Prepares workspaces for this operation.

        Args:
            inputs (list of GraphMultiStorage): Input data to this operation.

        log_forward class requires inputs to contain following keys.

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
            rm.cuda.culoge(self._inputs[gpu], self._outputs[gpu])


class log_forward_cpu(log_forward):

    def perform(self):
        self._outputs['cpu'] = np.log(self._inputs['cpu'])


class log_backward(operation):
    '''Log backward operation class.
    '''

    name = 'Log (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        '''Prepares workspaces for this operation.

        Args:
            inputs (list of GraphMultiStorage): Input data to this operation.

        log_backward class requires inputs to contain following keys.

        +-------+-----+------------------------------------+
        | Index | Key |              Role                  |
        +=======+=====+====================================+
        |   0   |  y  | Output of previous operation.      |
        +-------+-----+------------------------------------+
        '''

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
            self._outputs[gpu] = self._inputs[gpu] / self._fwd_inputs[gpu]


class log_backward_cpu(log_backward):

    def perform(self):
        self._outputs['cpu'] = self._inputs['cpu'] / self._fwd_inputs['cpu']


class LogElement(UserGraph):

    _name = 'Log Element'

    def __init__(self, previous_element=None):
        fwd_op = log_forward() if rm.is_cuda_active() else log_forward_cpu()
        bwd_ops = [log_backward(fwd_op) if rm.is_cuda_active() else log_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops,
                         previous_elements=previous_element)

@populate_graph
class Log(GraphFactory):
    '''A factory class of log function element.
    Log operation of the UserGraph object will call this factory class.

    Example:
        >>> import numpy as np
        >>> import renom.graph as rmg
        >>> 
        >>> x = np.arange(1, 7).reshape(2, 3)
        >>> layer = rmg.Log()
        >>> print(layer(x1))
        Log (F):
        [[0.        0.6931472 1.0986123]
        [1.3862944 1.609438  1.7917595]]
    '''

    def connect(self, x):
        return LogElement(previous_element=[x])


@populate_graph
@populate_basics
def log(self):
    '''A function style factory of log operation element.

    Args:
        self (UserGraph): Input array.

    For more information, please refer :py:class:`~renom.graph.basics.log_element.Log`.
    '''

    ret = Log()(self)
    return ret


UserGraph.log = log
