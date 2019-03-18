import renom as rm
import numpy as np
from renom.graph.core import operation, GraphMultiStorage, UserGraph


class placeholder_op(operation):

    name = 'Placeholder'
    roles = ['placeholder']

    def __init__(self, shape, gpus, identifier):
        shape = [1] + list(shape)
        self.identity = identifier

        outs = GraphMultiStorage(shape=shape, gpus=gpus)
        self._vars = {'y' : outs}

    def link(self, other_op):
        

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
