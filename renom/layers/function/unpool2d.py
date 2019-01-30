#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from renom.core import Node
from renom.layers.function.utils import im2col, col2im, transpose_out_size, tuplize
from renom.layers.function.pool2d import max_pool2d, average_pool2d
from renom.layers.function.unpoolnd import max_unpoolnd, average_unpoolnd
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import GPUValue, get_gpu
from .parameterized import Parametrized
from renom.config import precision

# A simple python object designed to hide the previous pool
# from ReNom


class SimpleContainer(object):
    def __init__(self, item):
        self._item = item


class max_unpool2d(Node):

    def __new__(cls, x, prev_pool):
        return cls.calc_value(x, prev_pool._item)

    _oper_cpu = max_unpoolnd._oper_cpu

    _oper_gpu = max_unpoolnd._oper_gpu

    _backward_cpu = max_unpoolnd._backward_cpu

    _backward_gpu = max_unpoolnd._backward_gpu


class average_unpool2d(Node):

    def __new__(cls, x, prev_pool):
        return cls.calc_value(x, prev_pool._item)

    _oper_cpu = average_unpoolnd._oper_cpu

    _oper_gpu = average_unpoolnd._oper_gpu

    _backward_cpu = average_unpoolnd._backward_cpu

    _backward_gpu = average_unpoolnd._backward_gpu


class MaxUnPool2d:
    '''Max unpooling function.
    Unpools an input in a network where a previous pooling has occured.

    Args:
        x (Node, np.ndarray):           The input to the unpooling method
        prev_pool (max_pool2d, None):   The previous pool to be unpooled. In the case of none,
                                        the model searches through the history for the previous layer.

    Note:
        The input shape requirement:
        ``x.shape == previous_pool.shape``

        The output shape will be:
        ``ret.shape == previous_pool.input.shape``

    '''

    def __init__(self):
        pass

    def __call__(self, x, prev_pool):
        return self.forward(x, SimpleContainer(prev_pool))

    def forward(self, x, prev_pool):
        return max_unpool2d(x, prev_pool)


class AverageUnPool2d:
    '''Average unpooling function.
    Unpools an input in a network where a previous pooling has occured.


    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> x = rm.Variable(np.random.rand(1, 3, 5, 5))
        >>> pool2d = rm.AveragePool2d(filter=3)
        >>> unpool2d = rm.AverageUnPool2d()
        >>>
        >>> print("Input", x.shape)
        Input (1, 3, 5, 5)
        >>>
        >>> h = pool2d(x)
        >>> print("Hidden layer", h.shape)
        Hidden layer (1, 3, 3, 3)
        >>>
        >>> z = unpool2d(h)
        >>> print("Output", z.shape)
        Last layer (1, 3, 5, 5)


    Note:
        The input shape requirement:
        ``x.shape == previous_pool.shape``

        The output shape will be:
        ``ret.shape == previous_pool.input.shape``

    '''

    def __init__(self):
        pass

    def __call__(self, x, prev_pool=None):
        p = x
        if prev_pool:
            assert isinstance(prev_pool, average_pool2d)
        while prev_pool is None:
            if isinstance(p, average_pool2d) and p.shape == x.shape:
                prev_pool = p
            else:
                try:
                    p = p.attrs._x
                except AttributeError:
                    raise Exception("Could not find previous 2D average pool")
        return self.forward(x, SimpleContainer(prev_pool))

    def forward(self, x, prev_pool=None):
        '''Forward operation of unpool 2d.
        The forward function of AverageUnPool2d layer accepts pool2d object.
        This layer unpools an input according to the given pool2d object.
        If pool2d is not given, this object find the previous pool2d object
        along the computational graph.

        Args:
            x (Node, ndarray): The input to the unpooling method
            prev_pool (average_pool2d): The previous pool to be unpooled. In the case of none, the model searches through the history for the previous layer.


        Returns:
            (average_unpool2d): Unpooled array.
        '''
        return average_unpool2d(x, prev_pool)
