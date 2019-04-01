#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

from functools import wraps

import numpy as np

import renom as rm
from .put_graph import put_graph
from .fetcher import *
from renom.graph import populate_graph


def indexer(index_func):
    @wraps(index_func)
    def new_index_func(self, *args, **kwargs):
        if self.indexed is True:
            raise AssertionError('The data should only be indexed once.')
        self.indexed = True
        return index_func(self, *args, **kwargs)
    return new_index_func


@populate_graph
class DataInput:
    '''A generic multi-source Input element.

    This method interfaces the normal Dataset/Distributor/Generator/etc. type
    interface for the ReNom graph.

    The DataInput class serves as an intermediary between the fetchers, which are responsible
    for producing data, and put graphs, which put the fetched data into the graphs.

    DataInput serves the basic idea of a pipeline, which is then finalized into a number of graphs
    that can be inserted into the ReNom graph.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> A = np.arange(6).reshape(3,2)
        >>> B = np.arange(6)[::-1].reshape(3,2)
        >>> A
        array([[0, 1],
               [2, 3],
               [4, 5]])
        >>> B
        array([[5, 4],
               [3, 2],
               [1, 0]])
        >>> x, y = rm.graph.DataInput([A, B]).index().batch(2).get_output_graphs()
        >>> try:
        ...     while(True):
        ...             print(x, y)
        ... except StopIteration:
        ...     print('Finished')
        ...
        Put Operation (0):
        [[0. 1.]
         [2. 3.]] Put Operation (1):
        [[5. 4.]
         [3. 2.]]
        Put Operation (0):
        [[4. 5.]] Put Operation (1):
        [[1. 0.]]
        Put Operation (0):
        [] Put Operation (1):
        []
        Finished
        >>>


    '''

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
        self.fetcher._reset()
        ret = []
        num_sources = self.fetcher.num_sources
        for source in range(num_sources):
            ret.append(put_graph(self.fetcher, source))
        if num_sources == 1:
            ret = ret[0]
        return ret

    @indexer
    def shuffle(self):
        '''Shuffles the input sources.

        This method counts as a form of indexer, transforming the indices
        of the input sources.
        '''
        self.fetcher = Shuffler(self.fetcher)
        return self

    @indexer
    def index(self):
        '''Inserts the input sources in order.

        This method counts as a form of indexer, inserting values from the
        sources in increasing order.
        '''
        self.fetcher = Indexer(self.fetcher)
        return self

    def batch(self, batch_size=32):
        '''Batches the received data.

        Each received value is gather until either the previous fetcher
        throws a StopIteration error or the batch_size is filled out. This
        method asserts the data was indexed.
        '''
        fetcher = self.fetcher
        if not (isinstance(fetcher, Indexer) or isinstance(fetcher, Shuffler)):
            self.shuffle()
        self.fetcher = Batcher(self.fetcher, batch_size)
        return self
