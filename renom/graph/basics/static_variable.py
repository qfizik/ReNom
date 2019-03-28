#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

import numpy as np

import renom as rm
from renom.graph.core import GraphMultiStorage, operational_element, UserGraph, operation
from renom.graph import populate_graph

class static_value(operation):

    name = 'Static Variable'
    roles = ['static']
    keyword = None

    def __init__(self, value, gpu_value=None):
        self._outputs = gpu_value
        self.gpus = gpu_value.gpus
        self._value_list = [value]
        self._vars = {'y': gpu_value}

    def setup(self, inputs):
        pass

    def perform(self):
        pass

    def _transfer_val(self, val):
        if rm.is_cuda_active():
            for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
                self._outputs[gpu].to_gpu(val)
        else:
            self._outputs['cpu'] = val

    def switch_source(self, id):
        new_val = self._value_list[id]
        self._transfer_val(new_val)

    @property
    def value(self):
        return self._outputs

    @value.setter
    def value(self, val):
        assert val.shape[1:] == self._outputs.shape[1:]
        assert val.shape[0] <= self._outputs.shape[0]
        self._outputs.shape[0].value = val.shape[0]
        self._transfer_val(val)

    def reset(self):
        pass


@populate_graph
class StaticVariable(UserGraph):

    _name = 'Static Element'

    def __init__(self, value, keyword=None, num_gpus=None):
        if value.dtype is not rm.precision:
            value = value.astype(rm.precision)
        if num_gpus is None:
            if GraphMultiStorage._gpus is None:
                num_gpus = 1
            else:
                num_gpus = GraphMultiStorage._gpus
        if rm.is_cuda_active():
            if isinstance(num_gpus, list):
                gpu_list = num_gpus
            else:
                gpu_list = [gpu for gpu in range(num_gpus)]
        else:
            gpu_list = 'cpu'
        val = GraphMultiStorage(shape=value.shape, gpus=gpu_list)
        if rm.is_cuda_active():
            for gpuv in val:
                gpuv.to_gpu(value)
        else:
            val['cpu'] = value
        self._value = val
        fwd_op = static_value(value, val)
        if keyword is not None:
            assert isinstance(keyword, str)
            fwd_op.keyword = keyword
        super().__init__(forward_operation=fwd_op)

    @property
    def value(self):
        return self._fwd._op.get_key('y')

    @value.setter
    def value(self, val):
        assert isinstance(val, np.ndarray)
        if self._value.shape == val.shape:
            if rm.is_cuda_active():
                for gpuv in self._value:
                    gpuv.to_gpu(val)
            else:
                self._value['cpu'] = val
        else:
            # TODO: FIX THIS
            raise NotImplementedError
