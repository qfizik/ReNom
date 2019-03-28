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
from renom.graph import populate_graph


class maxout_forward(operation):
    '''Maxout forward operation class.

    Args:
        slice_size (float): Coefficient used in maxout.
    '''

    name = 'Maxout (F)'

    def __init__(self, slice_size=1):
        self._sz = slice_size

    def setup(self, inputs):
        '''Prepares workspaces for this operation.

        Args:
            inputs (list of GraphMultiStorage): Input data to this operation.

        leaky_relu_forward class requires inputs to contain following keys.

        +-------+-----+--------------------------------+
        | Index | Key |              Role              |
        +=======+=====+================================+
        |   0   |  y  | Output of previous operation.  |
        +-------+-----+--------------------------------+
        '''

        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus

        axis = 1
        input_length = inputs.shape[axis]
        output_length = int(np.ceil(input_length / self._sz))
        out_shape = list(inputs.shape)
        out_shape[axis] = output_length
        outs = GraphMultiStorage(shape=tuple(out_shape), gpus=gpus)
        self._vars = {'y': outs}
        self._inputs = inputs
        self._outputs = outs

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            x = self._inputs[gpu].new_array()
            axis = 1
            input_length = self._inputs.shape[axis]
            slice_size = self._sz
            maxes = []
            for u in range(int(np.ceil(input_length / slice_size))):
                offset = u * slice_size
                maxes.append(np.amax(x[:, offset:offset + slice_size],
                                     axis=axis, keepdims=True))
            ret = np.concatenate(maxes, axis=axis)

            self._outputs[gpu].to_gpu(ret)


class maxout_forward_cpu(maxout_forward):

    def perform(self):
        x = self._inputs['cpu']
        axis = 1
        input_length = self._inputs.shape[axis]
        slice_size = self._sz
        maxes = []
        mask = np.zeros_like(x)
        for u in range(int(np.ceil(input_length / slice_size))):
            offset = u * slice_size
            args = np.argmax(x[:, offset:offset + slice_size], axis=axis)
            for b in range(len(x)):
                mask[b, args[b]] += 1
            maxes.append(np.amax(x[:, offset:offset + slice_size],
                                 axis=axis, keepdims=True))
        ret = np.concatenate(maxes, axis=axis)
        self._mask = mask

        self._outputs['cpu'] = ret


class maxout_backward(operation):
    '''Leaky relu backward operation class.

    Args:
        associated_forward (forward_operation): Corresponding forward operation.
    '''

    name = 'Maxout (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

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
        outs = GraphMultiStorage(shape=self._fwd_op._inputs.shape, gpus=gpus)
        self._vars = {'y': outs, id(self._fwd_op._inputs): outs}
        self._fwd_in = self._fwd_op._inputs
        self._inputs = inputs
        self._outputs = outs

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            dy = self._inputs[gpu].new_array()
            x = self._fwd_in[gpu].new_array()
            axis = 1
            input_length = dy.shape[1]
            slice_size = self._fwd_op._sz
            ret = np.zeros_like(x)
            for u in range(input_length):
                offset = u * slice_size
                args = np.argmax(x[:, offset:offset + slice_size], axis=axis)
                for b in range(len(x)):
                    ret[b, (args[b] + offset)] += dy[b, u]

            self._outputs[gpu].to_gpu(ret)


class maxout_backward_cpu(maxout_backward):

    def perform(self):
        dy = self._inputs['cpu']
        x = self._fwd_in['cpu']
        axis = 1
        input_length = dy.shape[1]
        slice_size = self._fwd_op._sz
        ret = np.zeros_like(x)
        for u in range(input_length):
            offset = u * slice_size
            args = np.argmax(x[:, offset:offset + slice_size], axis=axis)
            for b in range(len(x)):
                ret[b, (args[b] + offset)] += dy[b, u]

        self._outputs['cpu'] = ret


class MaxoutElement(UserGraph):

    def __init__(self, slice_size=1, previous_elements=None):
        fwd_op = maxout_forward(slice_size) if rm.is_cuda_active(
        ) else maxout_forward_cpu(slice_size)
        bwd_ops = [maxout_backward(fwd_op) if rm.is_cuda_active() else maxout_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


@populate_graph
class Maxout(GraphFactory):
    """A factory class of elu activation function element.

    Args:
        slice_size (int): The size of slices to perform maxout on.

    Example:
        >>> import numpy as np
        >>> import renom.graph as rmg
        >>>
        >>> x = np.array([-1, 0, 1]).reshape(1, -1)
        >>>
        >>> layer = rmg.Maxout()
        >>> layer(x)
        Maxout (F):
        [[-1.  0.  1.]]
        >>>
        >>> # Create element using function interface.
        >>> rmg.maxout(x)
        Maxout (F):
        [[-1.  0.  1.]]

    """

    def prepare(self, slice_size=1):
        self._sz = slice_size

    def connect(self, other):
        ret = MaxoutElement(self._sz, previous_elements=other)
        return ret


@populate_graph
def maxout(x, slice_size=1):
    return MaxoutElement(slice_size, previous_elements=[x])
