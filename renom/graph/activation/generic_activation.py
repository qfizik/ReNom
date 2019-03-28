#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

import renom as rm
from renom.graph import populate_graph


@populate_graph
class Activation:

    def __new__(cls, namestring, *args, **kwargs):
        '''Generic activation layer.

        This layer takes a string and produces an activation layer with default
        parameters, as defined in each class. Each string is viewed as lower
        capitalization, so ReLU, Relu, and relu will all produce the same result.

        The following strings are accepted and produced:
            relu -> Relu (No parameters)
            elu -> Elu (alpha=0.01)
            leaky relu -> LeakyRelu (slope=0.01)
            maxout -> Maxout (slice_size=1)
            selu -> Selu (No parameters)
            sigmoid -> Sigmoid (No parameters)
            softmax -> Softmax (No parameters)
            softplus -> Softplus (No parameters)
            tanh -> Tanh (No parameters)

        '''
        assert isinstance(namestring, str)
        rmg = rm.graph

        namestring = namestring.lower()

        if namestring == 'relu':
            ret = rmg.Relu(*args, **kwargs)
        elif namestring == 'elu':
            ret = rmg.Elu(*args, **kwargs)
        elif namestring == 'leaky relu':
            ret = rmg.LeakyRelu(*args, **kwargs)
        elif namestring == 'maxout':
            ret = rmg.Maxout(*args, **kwargs)
        elif namestring == 'selu':
            ret = rmg.Selu(*args, **kwargs)
        elif namestring == 'sigmoid':
            ret = rmg.Sigmoid(*args, **kwargs)
        elif namestring == 'softmax':
            ret = rmg.Softmax(*args, **kwargs)
        elif namestring == 'softplus':
            ret = rmg.Softplus(*args, **kwargs)
        elif namestring == 'tanh':
            ret = rmg.Tanh(*args, **kwargs)
        else:
            raise ValueError('Unknown name string')

        return ret
