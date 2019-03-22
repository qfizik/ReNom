import numpy as np
from renom import precision

class Fetcher:

    def __init__(self, prev, *args, **kwargs):
        self.prev = prev
        num_sources = prev.num_sources
        self.num_sources = num_sources
        self.outs = [None for source in range(num_sources)]
        self.start(num_sources, *args, **kwargs)
        for source in range(num_sources):
            self.prepare(source)
        self._reset()

    def start(self, num_sources):
        pass

    def prepare(self, source):
        pass

    def empty_out(self, source):
        prev = self.outs[source]
        new = np.array([], dtype=precision)
        new.resize(0, *prev.shape[1:])
        self.outs[source] = new

    def retrieve(self, source):
        return self.outs[source]

    def _reset(self):
        self.reset()
        self.prev._reset()

    def reset(self):
        pass

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

    def _reset(self):
        pass

    def __len__(self):
        return len(self.sources[0])

class Indexer(Fetcher):

    def start(self, num_sources):
        self.index = [0 for source in range(num_sources)]


    def prepare(self, source):
        if self.index[source] >= len(self):
            self.empty_out(source)
            raise StopIteration()
        prev = self.prev.retrieve(source)
        self.outs[source] = prev[self.index[source]].astype(precision)
        self.index[source] += 1

    def reset(self):
        self.index = [0 for source in range(self.num_sources)]


class Shuffler(Fetcher):

    def start(self, num_sources):
        self.index = [0 for source in range(num_sources)]
        self.perm = np.random.permutation(len(self))

    def prepare(self, source):
        if self.index[source] >= len(self):
            self.empty_out(source)
            raise StopIteration()
        prev = self.prev.retrieve(source)
        perm = self.perm
        index = self.index[source]
        self.outs[source] = prev[perm[index]].astype(precision)
        self.index[source] += 1

    def reset(self):
        self.index = [0 for source in range(self.num_sources)]
        self.perm = np.random.permutation(len(self))


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
            self.empty_out(source)
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
        length = int(np.ceil(len(self.prev) / self.batch_size))
        return length
