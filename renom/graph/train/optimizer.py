#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

import numpy as np

import renom as rm
from renom.graph.core import GraphMultiStorage
from renom.graph.train import initializer as init
from renom.graph import populate_graph


class optimizer_factory:

    def __init__(self):
        self._ops = {}
        self.args = ()
        self.kwargs = {}

    def _get_op(self, grad, out):
        key = "{}{}".format(id(grad), id(out))
        if key not in self._ops:
            self._ops[key] = self._create_op()
        return self._ops[key]

    def _create_op(self):
        if rm.is_cuda_active():
            ret = self._gpu_op(*self.args, **self.kwargs)
        else:
            ret = self._cpu_op(*self.args, **self.kwargs)
        return ret


@populate_graph
class Sgd(optimizer_factory):
    '''Stochastic Gradient Descent.

    Args:
        lr (float): Learning rate.
        momentum (float): Momentum coefficient of optimization.

    Example:
        >>> import numpy as np
        >>> import renom.graph as rmg
        >>> 
        >>> m = np.arange(4).reshape(2, 2)
        >>> layer = rmg.Dense(1)
        >>> 
        >>> out = layer(m)
        >>> weight = layer.params['w']
        >>> optimizer = rmg.Sgd()
        >>> 
        >>> print("Before update\n", weight)
        Before update
        Variable
        [[ 0.3224739]
        [-0.4718471]]
        >>> 
        >>> print("Before update\n", weight)
        >>> out.update(optimizer)
        >>> print("After update\n", weight)
        After update
        Variable
        [[ 0.31247392]
        [-0.4918471 ]]

    '''

    class _gpu_op:

        def __init__(self, learning_rate, momentum):
            self.learning_rate = learning_rate
            self.momentum = momentum
            self._outputs = None

        def setup(self, grad, val):
            if val is self._outputs:
                return
            self.gpus = grad.gpus
            self._dy = grad
            self._outputs = val
            self._run_avg = GraphMultiStorage(
                shape=grad.shape, gpus=grad.gpus, initializer=init.Constant(0))
            self._ndy = GraphMultiStorage(
                shape=grad.shape, gpus=grad.gpus, initializer=init.Constant(0))

        def update(self):
            for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
                rm.cuda.cu_optimizer_sgd(self.learning_rate, self.momentum,
                                         self._dy[gpu], self._run_avg[gpu], self._ndy[gpu], handle)
                rm.cuda.cusub(self._outputs[gpu], self._ndy[gpu], self._outputs[gpu], handle)
                self._run_avg[gpu] = self._ndy[gpu]

    class _cpu_op(_gpu_op):

        def update(self):
            dy = self._dy['cpu']
            cur = self._outputs['cpu']
            avg = self._run_avg['cpu']
            ret = (self.learning_rate * dy + avg * self.momentum)
            self._run_avg['cpu'] = ret
            self._outputs['cpu'] = cur - ret

    def __init__(self, learning_rate=0.01, momentum=0.4):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.args = (learning_rate, momentum)


@populate_graph
class Adagrad(optimizer_factory):
    '''Adaptive gradient algorithm. [Adagrad]_

    Args:
        lr (float): Learning rate.
        epsilon (float): Small number in the equation for avoiding zero division.

    .. [Adagrad] Duchi, J., Hazan, E., & Singer, Y. Adaptive Subgradient Methods for
        Online Learning and Stochastic Optimization. Journal of Machine Learning Research, 12, 2121–2159.
    '''

    class _gpu_op:

        def __init__(self, learning_rate, epsilon):
            self.learning_rate = learning_rate
            self.epsilon = epsilon
            self._outputs = None

        def setup(self, grad, val):
            if val is self._outputs:
                return
            self.gpus = grad.gpus
            self._dy = grad
            self._outputs = val
            self._prev = GraphMultiStorage(
                shape=grad.shape, gpus=grad.gpus, initializer=init.Constant(0))

        def update(self):
            for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
                ndy = self._outputs[gpu].empty_like_me()
                rm.cuda.cu_optimizer_adagrad(
                    self.learning_rate, self.epsilon, self._dy[gpu], self._prev[gpu], ndy, self._prev[gpu])
                self._outputs[gpu] -= ndy

    class _cpu_op(_gpu_op):

        def update(self):
            dy = self._dy['cpu']
            cur = self._outputs['cpu']
            pdy = self._prev['cpu']
            r = pdy + dy * dy
            ret = self.learning_rate * dy / (np.sqrt(r) + self.epsilon)
            self._outputs['cpu'] = cur - ret
            self._prev['cpu'] = r

    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        super().__init__()
        self._lr = learning_rate
        self._eps = epsilon
        self.args = (learning_rate, epsilon)


@populate_graph
class Adadelta(optimizer_factory):
    '''Adaptive gradient algorithm. [Adagrad]_

    Args:
        dr (float): Decay rate.
        epsilon (float): Small number in the equation for avoiding zero division.

    .. [Adagrad] Duchi, J., Hazan, E., & Singer, Y. Adaptive Subgradient Methods for
        Online Learning and Stochastic Optimization. Journal of Machine Learning Research, 12, 2121–2159.
    '''

    class _gpu_op:

        def __init__(self, dr, epsilon):
            self.dr = dr
            self.epsilon = epsilon
            self._outputs = None

        def setup(self, grad, val):
            if val is self._outputs:
                return
            self.gpus = grad.gpus
            self._dy = grad
            self._outputs = val
            self._pdy = GraphMultiStorage(
                shape=grad.shape, gpus=grad.gpus, initializer=init.Constant(0))
            self._pdx = GraphMultiStorage(
                shape=grad.shape, gpus=grad.gpus, initializer=init.Constant(0))

        def update(self):
            dr = self.dr
            for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
                ndy = self._outputs[gpu].empty_like_me()
                rm.cuda.cu_optimizer_adadelta(
                    dr, self.epsilon, self._pdy[gpu], self._pdx[gpu], self._dy[gpu], ndy)
                self._outputs[gpu] -= ndy

    class _cpu_op(_gpu_op):

        def update(self):
            dr = self.dr
            dy = self._dy['cpu']
            cur = self._outputs['cpu']
            pdy = self._pdy['cpu']
            pdx = self._pdx['cpu']
            E_squared_grad = dr * pdy + (1 - dr) * np.square(dy)
            dx = np.sqrt(pdx + self.epsilon) / np.sqrt(E_squared_grad + self.epsilon) * dy
            E_squared_x = dr * pdx + (1 - dr) * np.square(dx)
            ret = dx
            self._outputs['cpu'] = cur - ret
            self._pdx['cpu'] = E_squared_x
            self._pdy['cpu'] = E_squared_grad

    def __init__(self, dr=0.95, epsilon=1e-8):
        super().__init__()
        self._dr = dr
        self._eps = epsilon
        self.args = (dr, epsilon)


@populate_graph
class Adamax(optimizer_factory):
    '''Adamax optimizer for Adam using two running averages.

    Args:
        alpha (float): The effective learning rate.
        beta1 (float): Persistence for first running average.
        beta2 (float): Persistence for second running average.
        epsilon (float): Small value to avoid division by zero.
    '''
    class _gpu_op:

        def __init__(self, alpha, beta1, beta2, epsilon):
            self.alpha = alpha
            self.beta1 = beta1
            self.rb1 = beta1
            self.beta2 = beta2
            self.rb2 = beta2
            self.epsilon = epsilon
            self.time = 1
            self._outputs = None

        def setup(self, grad, val):
            if val is self._outputs:
                return
            self.gpus = grad.gpus
            self._dy = grad
            self._outputs = val
            self._mom1 = GraphMultiStorage(
                shape=grad.shape, gpus=grad.gpus, initializer=init.Constant(0))
            self._mom2 = GraphMultiStorage(
                shape=grad.shape, gpus=grad.gpus, initializer=init.Constant(0))

        def update(self):
            rb1 = self.rb1
            rb2 = self.rb2
            for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
                ndy = self._outputs[gpu].empty_like_me()
                rm.cuda.cu_optimizer_adamax(self.alpha, self.epsilon, (self.beta1, rb1),
                                            (self.beta2, rb2), self._mom1[gpu], self._mom2[gpu], self._dy[gpu], ndy)
                self._outputs[gpu] -= ndy
            self.rb1 = rb1 * self.beta1
            self.rb2 = rb2 * self.beta2

    class _cpu_op(_gpu_op):

        def update(self):
            alpha = self.alpha
            beta1 = self.rb1
            beta2 = self.rb2
            dy = self._dy['cpu']
            mom1 = self._mom1['cpu']
            mom2 = self._mom2['cpu']
            cur = self._outputs['cpu']

            new_mom1 = beta1 * mom1 + (1 - beta1) * dy
            new_mom2 = beta2 * mom2 + (1 - beta2) * dy ** 2
            mom1_est = new_mom1 / (1 - beta1)
            mom2_est = new_mom2 / (1 - beta2)
            ret = alpha * mom1_est / (np.sqrt(mom2_est) + self.epsilon)

            self._outputs['cpu'] = cur - ret
            self._mom1['cpu'] = new_mom1
            self._mom2['cpu'] = new_mom2
            self.rb1 = beta1 * self.beta1
            self.rb2 = beta2 * self.beta2

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.args = (alpha, beta1, beta2, epsilon)


@populate_graph
class Rmsprop(optimizer_factory):
    '''Rmsprop described by following formula. [Rmsprop]_

    .. math::

        m_{t+1} &=& gm_{t} + (1-g)\\nabla E^2 \\\\
        r_{t} &=& \\frac{lr}{\sqrt{m_{t+1}}+\epsilon} \\\\
        w_{t+1} &=& w_{t} - r_{t}\\nabla E

    Args:
        lr (float): Learning rate.
        g (float):
        epsilon (float): Small number in the equation for avoiding zero division.

    .. [Rmsprop] Nitish Srivastava, Kevin Swersky, Geoffrey Hinton. Neural Networks for Machine Learning.
    '''

    class _gpu_op:

        def __init__(self, lr, g, eps, r_avg):
            self.lr = lr
            self.g = g
            self.eps = eps
            self.r_avg = r_avg
            self._outputs = None

        def setup(self, grad, val):
            if val is self._outputs:
                return
            self.gpus = grad.gpus
            self._dy = grad
            self._outputs = val
            self._pmse = GraphMultiStorage(
                shape=grad.shape, gpus=grad.gpus, initializer=init.Constant(0))

        def update(self):
            for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
                lr, g, eps, r_avg = (self.lr, self.g, self.eps, self.r_avg)
                dy, r = (self._dy[gpu], self._pmse[gpu])
                ndy = dy.empty_like_me()
                rm.cuda.cu_optimizer_rmsprop(lr, eps, g, r_avg, dy, ndy, r)
                self._outputs[gpu] -= ndy

    class _cpu_op(_gpu_op):

        def update(self):
            lr, g, eps, r_avg = (self.lr, self.g, self.eps, self.r_avg)
            dy, pmse = (self._dy['cpu'], self._pmse['cpu'])

            r = g * pmse + (1 - g) * (dy ** 2)
            ret = lr * dy / (np.sqrt(r) + eps)

            self._outputs['cpu'] -= ret
            self._pmse['cpu'] = r

    def __init__(self, lr=0.001, g=0.9, epsilon=1e-8, running_average=1):
        super().__init__()
        self.args = (lr, g, epsilon, running_average)


@populate_graph
class Adam(optimizer_factory):
    '''Adaptive moment estimation described by following formula. [Adam]_

    .. math::

        m_{t+1} &=& bm_t + \\nabla E \\\\
        n_{t+1} &=& gn_t + \\nabla E^2 \\\\
        \\hat{m}_{t+1} &=& \\frac{m_{t+1}}{1-b^{t+1}} \\\\
        \\hat{n}_{t+1} &=& \\frac{n_{t+1}}{1-g^{t+1}} \\\\
        w_{t+1} &=& w_{t} - \\frac{\\alpha \hat{m}_{t+1}}{\sqrt{\hat{n}_{t+1}}+\epsilon}

    Args:
        lr (float): Learning rate.
        g (float): Coefficient
        b (float): Coefficient
        epsilon (float): Small number in the equation for avoiding zero division.


    .. [Adam] Diederik P. Kingma, Jimmy Ba. ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION(2014)
        https://arxiv.org/pdf/1412.6980.pdf
    '''

    class _gpu_op:

        CHECK_ZERO_VALUE = 100

        def __init__(self, alpha, beta1, beta2, epsilon):
            self.alpha = alpha
            self.beta1 = beta1
            self.rb1 = beta1
            self.beta2 = beta2
            self.rb2 = beta2
            self.epsilon = epsilon
            self.min = 2e-20
            self.time = 0
            self._outputs = None

        def setup(self, grad, val):
            if val is self._outputs:
                return
            self.gpus = grad.gpus
            self._dy = grad
            self._outputs = val
            self._mom1 = GraphMultiStorage(
                shape=grad.shape, gpus=grad.gpus, initializer=init.Constant(0))
            self._mom2 = GraphMultiStorage(
                shape=grad.shape, gpus=grad.gpus, initializer=init.Constant(0))

        def update(self):
            rb1 = self.rb1
            rb2 = self.rb2
            for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
                ndy = self._outputs[gpu].empty_like_me()
                rm.cuda.cu_optimizer_adam(self.alpha, self.epsilon, rb1, self.beta1, rb2, self.beta2, self.min,
                                          self.time % self.CHECK_ZERO_VALUE == 0, self._mom1[gpu],
                                          self._mom2[gpu], self._dy[gpu], ndy)
                self._outputs[gpu] = self._outputs[gpu] - ndy
            self.rb1 = rb1 * self.beta1
            self.rb2 = rb2 * self.beta2
            self.time += 1

    class _cpu_op(_gpu_op):

        def update(self):
            alpha = self.alpha
            beta1 = self.beta1
            beta2 = self.beta2
            powered_beta1 = self.rb1
            powered_beta2 = self.rb2
            dy = self._dy['cpu']
            mom1 = self._mom1['cpu']
            mom2 = self._mom2['cpu']
            cur = self._outputs['cpu']

            if self.time % self.CHECK_ZERO_VALUE == 0:
                mom1[np.abs(mom1) < self.min] = 0
                mom2[np.abs(mom2) < self.min] = 0

            lrt = alpha * np.sqrt(1 - powered_beta2) / (1 - powered_beta1)
            new_mom1 = beta1 * mom1 + (1 - beta1) * dy
            new_mom2 = beta2 * mom2 + (1 - beta2) * dy ** 2

            ret = lrt * new_mom1 / (np.sqrt(new_mom2) + self.epsilon)

            self._outputs['cpu'] = cur - ret
            self._mom1['cpu'] = new_mom1
            self._mom2['cpu'] = new_mom2
            self.rb1 = powered_beta1 * self.beta1
            self.rb2 = powered_beta2 * self.beta2
            self.time += 1

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.args = (alpha, beta1, beta2, epsilon)
