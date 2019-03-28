import numpy as np

import renom as rm
from renom import cuda
from renom.graph import populate_graph


class regularizer_factory:

    def __init__(self):
        self.__dict__['_ops'] = []
        self.args = ()
        self.kwargs = {}

    def create_op(self):
        if rm.is_cuda_active():
            ret = self.gpu_op(*self.args, **self.kwargs)
        else:
            ret = self.cpu_op(*self.args, **self.kwargs)
        self._ops.append(ret)
        return ret

    def __setattr__(self, name, val):
        for op in self._ops:
            setattr(op, name, val)
        self.__dict__[name] = val

@populate_graph
class L2(regularizer_factory):

    class gpu_op:

        def __init__(self, wd):
            self.wd = wd

        def setup(self, param, grad):
            self._param = param
            self._grad = grad
            self.gpus = param.gpus

        def apply(self):
            for gpu, handle in cuda.RenomHandlers(self.gpus):
                cuda.cu_l2_regularizer(self._param[gpu], self._grad[gpu], self.wd)

    class cpu_op(gpu_op):

        def apply(self):
            if self.wd == 0:
                pass
            else:
                self._grad['cpu'] += self._param['cpu'] * self.wd

    def __init__(self, wd=0.05):
        super().__init__()
        self.wd = wd
        self.args = (wd,)


class L1(regularizer_factory):

    class gpu_op:

        def __init__(self, wd):
            self.wd = wd

        def setup(self, param, grad):
            self._param = param
            self._grad = grad
            self.gpus = param.gpus

        def apply(self):
            raise NotImplementedError()

    class cpu_op(gpu_op):

        def apply(self):
            if self.wd == 0:
                return
            else:
                self._grad['cpu'] += np.sign(self._param['cpu']) * self.wd

    def __init__(self, wd=0.05):
        super().__init__()
        self._wd = wd
        self.args = (wd,)
