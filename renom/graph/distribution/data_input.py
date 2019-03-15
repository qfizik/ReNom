import renom as rm
import numpy as np
from renom.graph.core import operation, UserGraph, GraphMultiStorage


class data_sources:

    def __init__(self, sources):
        self.sources = sources
        self.num_sources = len(sources)
        self.cur_id = 0

    def retrieve(self, source):
        ret = self.sources[source]
        return ret

class indexer:

    def __init__(self, prev):
        self.prev = prev
        num_sources = prev.num_sources
        self.num_sources = num_sources
        self.index = [0 for source in range(num_sources)]

    def retrieve(self, source):
        prev = self.prev
        ret = prev.retrieve(source)[self.index[source]]
        self.index[source] += 1
        return ret

class shuffler:

    def __init__(self, prev):
        self.prev = prev
        self.perm = None
        self.index = 0

    def retrieve(self, source):
        prev = self.prev
        ret = prev.retrieve(source)
        perm = self.perm
        if perm is None:
            perm = np.random.permutation(len(ret) - 1)
            self.perm = perm
        ret = ret[perm[self.index]]
        self.index += 1
        return ret

class batcher:

    def __init__(self, prev, size):
        self.prev = prev
        self.size = size

    def retrieve(self, source):
        prev = self.prev
        size = self.size
        ret = []
        for i in range(size):
            ret.append(prev.retrieve(source))
        ret = np.array(ret)
        return ret

class put_op(operation):

    name = 'Put Operation'

    def __init__(self, fetcher, source):
        self.fetcher = fetcher
        self.source = source
        self.name = self.name + str(source)

    def setup(self, inputs):
        example = self.fetcher.retrieve(self.source)
        outs = GraphMultiStorage(shape = example.shape, gpus = 'cpu')
        self._vars = {'y' : outs}
        self._vars['y']['cpu'] = example

    def perform(self):
        pass
        #ret = self.fetcher.retrieve(self.source)
        #self._vars['y']['cpu'] = ret




class put_graph(UserGraph):


    def __init__(self, fetcher, source):
        fwd_op = put_op(fetcher, source)
        super().__init__(fwd_op)


class DataInput:

    def __init__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        self.fetcher = data_sources(inputs)
        self.sources = len(inputs)

    def get_output_graphs(self):
        ret = []
        print(self.fetcher)
        for source in range(self.sources):
            ret.append(put_graph(self.fetcher, source))
        return ret

    def shuffle(self):
        self.fetcher = shuffler(self.fetcher)
        return self

    def shuffle(self):
        self.fetcher = shuffler(self.fetcher)
        return self

    def range(self):
        self.fetcher = indexer(self.fetcher)
        return self

    def batch(self, batch_size = 32):
        self.fetcher = batcher(self.fetcher, batch_size)
        return self
