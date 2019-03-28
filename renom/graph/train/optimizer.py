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

    def get_op(self, grad, out):
        key = "{}{}".format(id(grad), id(out))
        if key not in self._ops:
            self._ops[key] = self.create_op()
        return self._ops[key]

    def create_op(self):
        if rm.is_cuda_active():
            ret = self.gpu_op(*self.args, **self.kwargs)
        else:
            ret = self.cpu_op(*self.args, **self.kwargs)
        return ret


T = True
F = False


@populate_graph
class Sgd(optimizer_factory):

    class gpu_op:

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

    class cpu_op(gpu_op):

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


class Adagrad(optimizer_factory):

    class gpu_op:

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

    class cpu_op(gpu_op):

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


class Adadelta(optimizer_factory):

    class gpu_op:

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

    class cpu_op(gpu_op):

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


class Adamax(optimizer_factory):

    class gpu_op:

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

    class cpu_op(gpu_op):

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


class Rmsprop(optimizer_factory):

    class gpu_op:

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
            self._prav = GraphMultiStorage(
                shape=grad.shape, gpus=grad.gpus, initializer=init.Constant(0))

        def update(self):
            for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
                lr, g, eps, r_avg = (self.lr, self.g, self.eps, self.r_avg)
                dy, r, k = (self._dy[gpu], self._pmse[gpu], self._prav[gpu])
                ndy = dy.empty_like_me()
                rm.cuda.cu_optimizer_rmsprop(lr, eps, g, r_avg, dy, k, ndy, r)
                self._outputs[gpu] -= ndy

    class cpu_op(gpu_op):

        def update(self):
            lr, g, eps, r_avg = (self.lr, self.g, self.eps, self.r_avg)
            dy, pmse, prav = (self._dy['cpu'], self._pmse['cpu'], self._prav['cpu'])

            r = g * pmse + (1 - g) * (dy ** 2)
            k = r_avg * prav + (1 - r_avg) * dy
            v = (r - k**2)
            v[v < 0] = 0
            ret = lr * dy / np.sqrt(v + eps)

            self._outputs['cpu'] -= ret
            self._pmse['cpu'] = r
            self._prav['cpu'] = k

    def __init__(self, lr=0.001, g=0.9, epsilon=1e-8, running_average=1):
        super().__init__()
        self.args = (lr, g, epsilon, running_average)


class Adam(optimizer_factory):

    class gpu_op:

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

    class cpu_op(gpu_op):

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
