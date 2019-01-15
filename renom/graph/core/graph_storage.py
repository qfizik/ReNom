import numpy as np
import renom as rm


class shared_val:
    '''
          This object is responsible for allowing each GraphMultiStorage to
          share the definition of their batch sizes.

          The shape of a GraphMultiStorage object is (s, *i) where s is a shared_val
          and *i is any number of integers.

          Each graph will thus have exactly one shared_val object, which determines the
          batch size of the current tree, allowing us to only worry about setting this
          value once while updating the entire tree.
    '''
    def __new__(cls, val, m=None):
        return super().__new__(cls)

    def __init__(self, val):
        self._val = val
        self._max = val

    @property
    def value(self):
        return self._val

    @value.setter
    def value(self, val):
        if val <= self._max:
            self._val = val
        else:
            raise AttributeError('Setting val above initial value')

    def __int__(self):
        return self._val

    def __float__(self):
        return float(self._val)

    def __mul__(self, other):
        return self._val * int(other)

    def __add__(self, other):
        return self._val + int(other)

    def __eq__(self, other):
        return self._val == int(other)

    def __index__(self):
        return self._val

    def __repr__(self):
        return self._val.__repr__()

    def __lt__(self, other):
        return self._val < int(other)

    def __gt__(self, other):
        return self._val > int(other)

    def __neg__(self):
        return -self._val

    def __floordiv__(self, other):
        return self._val // int(other)


class GraphMultiStorage:
    '''
          The storage class of the graph. The graph is constructed through a brick and
          mortar construction, type with the GraphMultiStorage being the party responsible
          for keeping operations tied together indirectly.

          The key points of the GraphMultiStorage class is that it allows device agnosticism
          through only indirectly implementing the storage. The actual data is contained in
          the GPUValue objects.

          When interfacing data for external devices, indexing the object will retrieve
          the GPUValue defined on the indexed device.
          When interfacing data for the cpu however, the storage should be indexed using
          the 'cpu' special string.
    '''
    ready = False

    def __init__(self, shape=None, gpus=None, initializer=None, ptrs=None):
        if self.ready is True:
            assert self.shape == shape
            return
        if shape is None:
            return
        if gpus is None:
            gpus = [0]
        self.gpus = gpus
        self._gpuvalues = {}
        self._initializer = initializer
        self._finished_setup = False
        self._ptrs = ptrs
        self.shape = shape
        if ptrs is not None:
            assert isinstance(ptrs, GraphMultiStorage)
            shp = list(self.shape)
            shp[0] = ptrs.shape[0]
            self.shape = shp
        self._create_values()
        self.ready = True

    def _create_values(self):
        self._finished_setup = True
        if self.gpus == 'cpu':
            if self._initializer is not None:
                self['cpu'] = self._initializer(self.shape)
            else:
                self['cpu'] = np.empty(self.shape)
            return

        for gpu in self.gpus:
            with rm.cuda.RenomHandler(gpu):
                arr = None
                if self._initializer is not None:
                    arr = self._initializer(self.shape)
                meminfo = rm.cuda.cuGetMemInfo()
                assert np.prod(self.shape) * np.dtype(rm.precision).itemsize <= meminfo[0]
                self[gpu] = rm.GPUValue(array=arr, shape=self.shape,
                                        ptr=self._ptrs[gpu]._ptr if self._ptrs is not None else None)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, val):
        if not isinstance(val[0], shared_val):
            new_val = shared_val(val[0])
            lst = list(val)
            lst[0] = new_val
            val = tuple(lst)
        assert isinstance(val[0], shared_val)
        self._shape = tuple(val)

    @property
    def gpus(self):
        return self._gpus

    @gpus.setter
    def gpus(self, val):
        assert isinstance(val, list) or isinstance(val, str)
        if isinstance(val, str):
            assert val == 'cpu'
        self._gpus = val

    def __iter__(self):
        for key in self._gpuvalues.keys():
            yield self[key]

    def __len__(self):
        if isinstance(self._gpus, str):
            return 0
        return len(self._gpus)

    def __getitem__(self, index):
        if self._ptrs is not None and self.gpus == 'cpu':
            return self._ptrs[index].reshape(self.shape)
        return self._gpuvalues[index]

    def __setitem__(self, index, value):
        if self.gpus == 'cpu':
            self._gpuvalues[index] = value
            return
        if tuple(value.shape) != tuple(self.shape):
            print(value.shape, self.shape)
            raise AssertionError('Setting value without same shape')
        value.shape = self.shape
        self._gpuvalues[index] = value

    def __repr__(self):
        assert self._finished_setup is True
        k = self._gpuvalues.keys().__iter__().__next__()
        return self._gpuvalues[k].__str__()

    def as_ndarray(self):
        if not rm.is_cuda_active():
            return self._gpuvalues['cpu']
        ret = 0
        for gpu in self.gpus:
            ret += self._gpuvalues[gpu].new_array()
        return ret
