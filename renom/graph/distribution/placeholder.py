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
        vars = other_op.get_output_signature()
        ins = vars['y']
        assert ins.shape[1:] == self._out.shape[1:]
        self._out.shape = ins.shape
        self._ins = ins
        vars['y'] = self._out

    def setup(self, inputs):
        pass

    def perform(self):
        pass

class Placeholder(UserGraph):


    def __init__(self, shape, num_gpus=1):
        if rm.is_cuda_active():
            gpus = [gpu for gpu in range(num_gpus)]
        else:
            gpus = 'cpu'
        fwd_op = placeholder_op(shape, gpus, id(self))
        super().__init__(fwd_op)
