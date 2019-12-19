usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
from renom.core import BinOp, Node, get_gpu
from renom.cuda import cuda as cu


class mean_squared_error(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        assert rhs.ndim > 1, "Input arrays must have no less than 2 dimension."
        N = len(lhs)
        return np.sum((lhs - rhs) ** 2) / (N * 2)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        assert rhs.ndim > 1, "Input arrays must have no less than 2 dimension."
        N = len(lhs)
        return cu.cusum(get_gpu((get_gpu(lhs) - get_gpu(rhs)) ** 2)) / (N * 2)

    def _backward_cpu(self, context, dy, **kwargs):
        sub = self.attrs._lhs - self.attrs._rhs
        if isinstance(self.attrs._lhs, Node):
            N = len(self.attrs._lhs)
            self.attrs._lhs._update_diff(context, sub * dy / N, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            N = len(self.attrs._lhs)
            sub = get_gpu(self.attrs._lhs) - get_gpu(self.attrs._rhs)
            self.attrs._lhs._update_diff(context, sub * get_gpu(dy) / N, **kwargs)


class MeanSquaredError(object):
    """This function evaluates the loss between the target ``y``
    and the input ``x`` using mean squared error.

    .. math::
        E(x) = \\frac{1}{2N}\sum_{n}^{N}\sum_{k}^{K}(x_{nk}-y_{nk})^2

    :math:`N` is batch size.

    Args:
        x (ndarray,Node): Input array.
        y (ndarray,Node): Target array.

    Raises:
        AssertionError: An assertion error will be raised if the given tensor dimension is less than 2.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>>
        >>> x = np.array([[1, 1]])
        >>> y = np.array([[-1, -1]])
        >>> print(x.shape, y.shape)
        ((1, 2), (1, 2))
        >>> loss = rm.mean_squared_error(x, y)
        >>> print(loss)
        mean_squared_error(4.0)
        >>>
        >>> # Also you can call this function with alias.
        >>> loss = rm.mse(x, y)
        >>> print(loss)
        mean_squared_error(4.0)

    """

    def __call__(self, x, y):
        return mean_squared_error(x, y)
