import renom as rm
import numpy as np
from renom.graph.core import operation, UserGraph, GraphMultiStorage

class Fetcher:

    def __init__(self, prev, *args, **kwargs):
        self.prev = prev
        num_sources = prev.num_sources
        self.num_sources = num_sources
        self.outs = [None for source in range(num_sources)]
        self.start(num_sources, *args, **kwargs)
        for source in range(num_sources):
            self.prepare(source)

    def start(self, num_sources):
        pass

    def prepare(self, source):
        pass

    def retrieve(self, source):
        return self.outs[source]

    def reset(self):
        self.prev.reset()

    def __len__(self):
        return len(self.prev)



class DataSources(Fetcher):

    def __init__(self, sources):
        assert len(sources) > 0
        self.sources = sources
        num_sources = len(sources)
        self.num_sources = num_sources
        if sources[0] is None:
            assert all(source is None for source in sources)
        else:
            lengths = [len(source) for source in sources]
            assert all(lengths[i] == lengths[i+1] for i in range(num_sources - 1))

    def retrieve(self, source):
        ret = self.sources[source]
        return ret

    def __len__(self):
        return len(self.sources[0])

class Indexer(Fetcher):

    def start(self, num_sources):
        self.index = [0 for source in range(num_sources)]


    def prepare(self, source):
        if self.index[source] >= len(self):
            raise StopIteration()
        prev = self.prev.retrieve(source)
        self.outs[source] = prev[self.index[source]]
        self.index[source] += 1

    def reset(self):
        self.index = [0 for source in range(self.num_sources)]


class Shuffler(Fetcher):

    def start(self, num_sources):
        self.index = [0 for source in range(num_sources)]
        length = len(self.prev)
        self.perm = np.random.permutation(length)

    def prepare(self, source):
        if self.index[source] >= self.lengths[source]:
            raise StopIteration()
        prev = self.prev.retrieve(source)
        perm = self.perms[source]
        index = self.index[source]
        self.outs[source] = prev[perm[index]]
        self.index += 1

class Batcher(Fetcher):

    def start(self, source, size):
        self.batch_size = size

    def prepare(self, source):
        prev = self.prev
        ret = []
        try:
            prev.prepare(source)
            prev_val = prev.retrieve(source)
            ret.append(prev_val)
        except StopIteration as e:
            raise e
        size = self.batch_size
        for i in range(size-1):
            try:
                prev.prepare(source)
                prev_val = prev.retrieve(source)
                ret.append(prev_val)
            except StopIteration as e:
                break
        ret = np.array(ret)
        self.outs[source] = ret

    def __len__(self):
        length = len(self.prev) // self.batch_size
        return length

class put_op(operation):

    name = 'Put Operation'
    roles = ['input']

    def __init__(self, fetcher, source):
        self.fetcher = fetcher
        self.source = source
        self.name = self.name + ' ({})'.format(self.source)
        self.finished = False
        example = self.fetcher.retrieve(self.source)
        outs = GraphMultiStorage(shape = example.shape, gpus = 'cpu')
        self._vars = {'y' : outs}
        self._vars['y']['cpu'] = example

    def setup(self, inputs):
        pass

    def __len__(self):
        return len(self.fetcher)

    def reset(self):
        self.fetcher.reset()

    def perform(self):
        try:
            self.fetcher.prepare(self.source)
            ret = self.fetcher.retrieve(self.source)
            self._vars['y']['cpu'] = ret
        except StopIteration as e:
            raise e




class put_graph(UserGraph):


    def __init__(self, fetcher, source):
        fwd_op = put_op(fetcher, source)
        super().__init__(fwd_op)


class DataInput:

    def __init__(self, inputs):
        self.inputs = inputs
        self.fetcher = DataSources(inputs)
        self.num_sources = len(inputs)

    def get_output_graphs(self):
        ret = []
        for source in range(self.num_sources):
            ret.append(put_graph(self.fetcher, source))
        return ret

    def shuffle(self):
        self.fetcher = Shuffler(self.fetcher)
        return self

    def index(self):
        self.fetcher = Indexer(self.fetcher)
        return self

    def batch(self, batch_size = 32):
        self.fetcher = Batcher(self.fetcher, batch_size)
        return self
