import numpy as np
import renom as rm
from renom.graph.core import operation, UserGraph, GraphMultiStorage

class put_op(operation):

    name = 'Put Operation'
    roles = ['input']

    def __init__(self, fetcher, source):
        self.fetcher = fetcher
        self.source = source
        self.name = self.name + ' ({})'.format(self.source)
        example = self.fetcher.retrieve(self.source)
        outs = GraphMultiStorage(shape = example.shape, gpus = 'cpu')
        self._vars = {'y' : outs}
        self._vars['y']['cpu'] = example
        self.reset()

    def setup(self, inputs):
        pass

    def __len__(self):
        return len(self.fetcher) + 1

    def reset(self):
        if self.source == 0:
            self.fetcher._reset()
        self._finished = False

    def perform(self):
        if self._finished is True:
            raise StopIteration()
        try:
            self.fetcher.prepare(self.source)
        except StopIteration as e:
            self._finished = True
        ret = self.fetcher.retrieve(self.source)
        self._vars['y']['cpu'] = ret
        self._vars['y'].shape[0].value = ret.shape[0]




class put_graph(UserGraph):


    def __init__(self, fetcher, source):
        fwd_op = put_op(fetcher, source)
        super().__init__(fwd_op)

    def reset(self):
        self._fwd._op.reset()
