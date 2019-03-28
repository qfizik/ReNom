#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

import numpy as np

import renom as rm
from renom.graph.core import operation, operational_element, UserGraph, \
    GraphMultiStorage, GraphFactory
from renom.graph import populate_graph
from renom.graph.basics import populate_basics


class get_item_forward(operation):

    name = 'Get Item (F)'

    def __init__(self, index):
        self._index = index

    def setup(self, inputs):
        a = inputs[0]['y']
        self.gpus = a.gpus
        self._a = a
        tmp = np.empty(a.shape)[self._index]
        self._b = GraphMultiStorage(shape=tmp.shape, gpus=self.gpus)
        self._vars = {'a': a, 'b': self._b, 'y': self._b}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            a = self._a[gpu]
            self._b[gpu] = a[self._index]


class get_item_forward_cpu(get_item_forward):

    def perform(self):
        a = self._a['cpu']
        self._b['cpu'] = a[self._index]


def _backward_cpu(self, context, dy, **kwargs):
    if isinstance(self.attrs._lhs, Node):
        zero = np.zeros_like(to_value(self.attrs._lhs))
        np.add.at(zero, self.attrs._rhs, to_value(dy))
        self.attrs._lhs._update_diff(context, zero, **kwargs)


def _backward_gpu(self, context, dy, **kwargs):
    if isinstance(self.attrs._lhs, Node):
        if self._is_advanced_indexing(self.attrs._lhs, self.attrs._rhs):
            self._backward_cpu(context, to_value(dy), **kwargs)
        else:
            zero = get_gpu(self.attrs._lhs).zeros_like_me()
            zero[self.attrs._rhs] = dy
            self.attrs._lhs._update_diff(context, zero, **kwargs)


def _is_advanced_indexing(index):
    if isinstance(index, (int, slice, type(None), type(Ellipsis))):
        return False
    elif isinstance(index, tuple):
        if all([isinstance(o, (int, slice, type(None), type(Ellipsis))) for o in index]):
            return False
    elif isinstance(index, np.ndarray):
        if index.dtype == np.bool:
            return False
    return True


class get_item_backward(operation):

    name = 'Get Item (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus
        self._index = self._fwd_op._index
        out_shape = self._fwd_op._a.shape
        outs = GraphMultiStorage(shape=out_shape, gpus=self.gpus)
        self._inputs = inputs
        self._outputs = outs
        self._vars = {'y': outs, id(self._fwd_op._a): outs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            if _is_advanced_indexing(self._index):
                dy = self._inputs[gpu].new_array()
                zero = np.zeros(self._outputs.shape)
                np.add.at(zero, self._index, dy)
                self._outputs[gpu].to_gpu(zero)
            else:
                dy = self._inputs[gpu]
                zero = self._outputs[gpu].zeros_like_me()
                zero[self._index] = dy
                self._outputs[gpu] = zero


class get_item_backward_cpu(get_item_backward):

    def perform(self):
        dy = self._inputs['cpu']
        zero = np.zeros(self._outputs.shape)
        np.add.at(zero, self._index, dy)
        self._outputs['cpu'] = zero


class GetItemElement(UserGraph):

    _name = 'Add Element'

    def __init__(self, index, previous_elements=None):

        fwd_op = get_item_forward(index) if rm.is_cuda_active() else get_item_forward_cpu(index)
        bwd_ops = [get_item_backward(fwd_op) if rm.is_cuda_active()
                   else get_item_backward_cpu(fwd_op)]
        super().__init__(fwd_op, bwd_ops, previous_elements)

@populate_graph
@populate_basics
class GetItem(GraphFactory):
    """GetItem
    """

    def connect(self, x, index):
        return GetItemElement(index, previous_elements=[x])


def _get_item(self, index):
    ret = GetItem()(index, self)
    return ret


UserGraph.__getitem__ = _get_item
