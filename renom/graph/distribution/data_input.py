import renom as rm
import numpy as np
from .put_graph import put_graph
from functools import wraps
from .fetcher import *


def indexer(index_func):
    @wraps(index_func)
    def new_index_func(self, *args, **kwargs):
        if self.indexed is True:
            raise AssertionError('The data should only be indexed once.')
        self.indexed = True
        return index_func(self, *args, **kwargs)
    return new_index_func


class DataInput:

    def __init__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        self.inputs = inputs
        self.fetcher = DataSources(inputs)
        self._orig_num_sources = len(inputs)
        self.indexed = False

    def get_output_graphs(self):
        assert self.indexed is True, 'The input sources must be indexed in ' \
                    + 'some way before being converted to graphs!'
        ret = []
        num_sources = self.fetcher.num_sources
        for source in range(num_sources):
            ret.append(put_graph(self.fetcher, source))
        if num_sources == 1:
            ret = ret[0]
        return ret

    @indexer
    def shuffle(self):
        self.fetcher = Shuffler(self.fetcher)
        return self

    @indexer
    def index(self):
        self.fetcher = Indexer(self.fetcher)
        return self

    def batch(self, batch_size = 32):
        fetcher = self.fetcher
        if not (isinstance(fetcher, Indexer) or isinstance(fetcher, Shuffler)):
            self.shuffle()
        self.fetcher = Batcher(self.fetcher, batch_size)
        return self
