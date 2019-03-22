import renom as rm
import numpy as np
from renom.graph.core import operation, GraphMultiStorage, UserGraph


class placeholder_op(operation):

    name = 'Placeholder'
    roles = ['placeholder']

    def __init__(self, shape, gpus, identifier):
        shape = [1] + list(shape)
        self.identity = identifier
        self._other = None

        outs = GraphMultiStorage(shape=shape, gpus=gpus)
        self._out = outs
        self._vars = {'y' : outs}
        self.gpus = gpus

    def link(self, other_op):
        self._other = other_op
        ins = other_op.get_output_signature()['y']
        assert ins.shape[1:] == self._out.shape[1:]
        self._out.shape = ins.shape
        self._ins = ins

    def setup(self, inputs):
        pass

    def perform(self):
        if self._other is not None:
            if rm.is_cuda_active():
                for gpu in self.gpus:
                    self._out[gpu] = self._ins[gpu]
            else:
                for gpu in self.gpus:
                    self._out['cpu'] = self._ins['cpu']
        else:
            raise AttributeError('Placeholder has not yet been linked.')

class Placeholder(UserGraph):


    def __init__(self, shape, num_gpus=1):
        if rm.is_cuda_active():
            gpus = [gpu for gpu in range(num_gpus)]
        else:
            gpus = 'cpu'
        fwd_op = placeholder_op(shape, gpus, id(self))
        super().__init__(fwd_op)
