import renom as rm
import numpy as np

class regularizer_factory:

    def __init__(self):
        self.args = ()
        self.kwargs = {}

    def create_op(self):
        if rm.is_cuda_active():
            return self.gpu_op(*self.args, **self.kwargs)
        else:
            return self.cpu_op(*self.args, **self.kwargs)

class l2_regularizer(regularizer_factory):

    class gpu_op:

        def __init__(self, wd):
            self.wd = wd

        def setup(self, param, grad):
            self._param = param
            self._grad = grad

        def apply(self):
            pass

    class cpu_op(gpu_op):

        def apply(self):
            print('Called')
            self._grad['cpu'] += self._param['cpu'] * self.wd

    def __init__(self, wd=0.05):
        super().__init__()
        self._wd = wd
        self.args = (wd,)
