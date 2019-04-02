#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

import numpy as np

import renom as rm
from renom.graph.core import operation, UserGraph, GraphMultiStorage


class put_op(operation):

    name = 'Put Operation'
    roles = ['input']

    def __init__(self, fetcher, source, gpus):
        self.fetcher = fetcher
        self.source = source
        self.name = self.name + ' ({})'.format(self.source)
        self.gpus = gpus
        example = self.fetcher.retrieve(self.source)
        outs = GraphMultiStorage(shape=example.shape, gpus=gpus)
        self._vars = {'y': outs}
        self._vars['y']['cpu'] = example
        self.reset()

    def setup(self, inputs):
        pass

    def __len__(self):
        return len(self.fetcher)

    def reset(self):
        if self.source == 0:
            self.fetcher._reset()
        self._finished = False

    def perform(self):
        if self._finished is True:
            raise StopIteration()
        try:
            self.fetcher.prepare(self.source)
        except StopIteration:
            raise StopIteration()
            self._finished = True
        ret = self.fetcher.retrieve(self.source)
        self._vars['y'].shape[0].value = ret.shape[0]
        if rm.is_cuda_active():
            for gpu in self.gpus:
                self._vars['y'][gpu] = rm.GPUValue(ret)
        else:
            self._vars['y']['cpu'] = ret


class put_graph(UserGraph):

    def __init__(self, fetcher, source, gpus=1):
        if rm.is_cuda_active():
            gpus = [gpu for gpu in range(gpus)]
        else:
            gpus = 'cpu'
        fwd_op = put_op(fetcher, source, gpus)
        super().__init__(fwd_op)

    def reset(self):
        self._fwd._op.reset()
