#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
from renom.core import Node, get_gpu
from renom.layers.activation import softmax
from renom.cuda import cuda as cu


class softmax_cross_entropy(Node):

    def __new__(cls, lhs, rhs):
        assert rhs.ndim > 1, "Input arrays must have no less than 2 dimension."
        return cls.calc_value(lhs, rhs)

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        N = len(lhs)
        z = softmax(lhs)
        loss = -np.sum(rhs * np.log(z + 1e-8)) / N
        ret = cls._create_node(loss)
        ret.attrs._z = z
        ret.attrs._lhs = lhs
        ret.attrs._rhs = rhs
        return ret

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        N = lhs.shape[0]
        z = softmax(lhs)
        tmp1 = get_gpu(lhs).empty_like_me()
        cu.cucross_entropy(get_gpu(z), get_gpu(rhs), get_gpu(tmp1))
        loss = -cu.cusum(get_gpu(tmp1)) / N
        ret = cls._create_node(loss)
        ret.attrs._z = z
        ret.attrs._lhs = lhs
        ret.attrs._rhs = rhs
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            N = len(self.attrs._lhs)
            sub = self.attrs._z - self.attrs._rhs
            self.attrs._lhs._update_diff(context, sub * dy / N, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            N = len(self.attrs._lhs)
            sub = get_gpu(self.attrs._z) - get_gpu(self.attrs._rhs)
            self.attrs._lhs._update_diff(context, sub * get_gpu(dy) / N, **kwargs)


class SoftmaxCrossEntropy(object):
    """This function evaluates the loss between target ``y`` and output
    of softmax activation ``z`` using cross entropy.

    .. math::
        z_{nk} &= \\frac{\exp(x_{nk})}{\sum_{j=1}^{K}\exp(x_{nj})} \\\\
        E(x) &= -\\frac{1}{N}\sum_{n}^{N}\sum_{k}^{K}y_{nk}\log(z_{nk})

    Args:
        x (ndarray,Node): Input array.
        y (ndarray,Node): Target array.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> 
        >>> x = np.array([[0, 1]])
        >>> y = np.array([[1, 0]])
        >>> loss_func = rm.SoftmaxCrossEntropy()
        >>> loss = loss_func(x, y)
        >>> print(loss)
        1.31326162815094
        >>> loss = rm.softmax_cross_entropy(x, y)
        >>> print(loss)
        1.31326162815094
        >>>
        >>> # You can call this function with alias.
        >>> loss = rm.smce(x, y)
        >>> print(loss)
        1.31326162815094


    Raises:
        AssertionError: An assertion error will be raised if the given tensor dimension is less than 2.
    """

    def __call__(self, lhs, rhs):
        return softmax_cross_entropy(lhs, rhs)
