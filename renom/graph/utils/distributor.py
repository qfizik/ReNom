import numpy as np
import renom as rm
from renom.graph.core import operational_element, UserGraph, operation, GraphMultiStorage


class dispatch(operation):
    '''
    Dispatch class, responsible for feeding input data to the graph model.

    Performing this operation produces the next output value until the requested batch size cannot be fulfilled.
    Once the batch size can no longer be fulfilled with the given input source,
    the operation produces a StopIteration exception, which the user is requested to catch.

    To run through the input source again, the reset method should be called,
    which will handle reinitializing the internal states.
    '''
    name = 'Data Dispatcher'
    roles = ['input']

    def __init__(self, value, batch_size=128, num_gpus=1, shuffle=True):
        self._value_list = value
        if len(value) > 1:
            self._has_validation_data = True
        else:
            self._has_validation_data = False
        self._value = value[0]
        self._batch_num = 0
        self._batch_size = batch_size
        out_shape = [batch_size]
        out_shape.extend(self._value.shape[1:])
        self._num_gpus = num_gpus
        self.gpus = [gpu for gpu in range(num_gpus)] if rm.is_cuda_active() else 'cpu'
        self._outputs = GraphMultiStorage(shape=out_shape, gpus=self.gpus)
        self._vars = {'y': self._outputs}
        self._finished = False
        self._shuffle = shuffle
        self._perm = np.random.permutation(
            len(self._value)) if self._shuffle else np.arange(len(self._value))
        self._attached = None
        self._master = None

    def setup(self, inputs):
        self._batch_vars = [v.shape[0] for v in self._outputs]

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_val):
        self._value = new_val
        self.reset()

    def perform(self):
        if self._finished:
            raise StopIteration
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            # handle.wait()
            cur_slice = slice(self._batch_num * self._batch_size,
                              (1 + self._batch_num) * self._batch_size)
            arr = self._value[self._perm[cur_slice]]
            self._outputs[gpu].shape[0].value = len(arr)
            assert self._outputs[gpu].shape == arr.shape
            if len(arr) < self._batch_size:
                self._finished = True
            pin = handle.getPinnedMemory(arr)
            assert pin.shape == self._outputs[gpu].shape
            self._outputs[gpu].to_gpu(pin)
            self._batch_num += 1

    def switch_source(self, id):
        if self._master is not None:
            return
        assert self._attached is not None
        other = self._attached
        self._value = self._value_list[id]
        other._value = other._value_list[id]
        self.reset()

    def change_input(self, new_values):
        self._value_list = [new_values]
        self._value = new_values
        self.reset()

    def attach(self, other):
        assert isinstance(other, dispatch)
        assert self._value.shape[0] == other._value.shape[0]
        self._attached = other
        other._master = self
        self._perm = other._perm

    def reset(self):
        if self._master is not None:
            return
        assert self._attached is not None
        other = self._attached
        self._batch_num = 0
        self._finished = False
        other._batch_num = 0
        other._finished = False
        self._perm = np.random.permutation(
            len(self._value)) if self._shuffle else np.arange(len(self._value))
        other._perm = self._perm

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self._value) / self._batch_size))


class dispatch_cpu(dispatch):

    def perform(self):
        if self._finished:
            raise StopIteration
        cur_slice = slice(self._batch_num * self._batch_size,
                          (1 + self._batch_num) * self._batch_size)
        arr = self._value[self._perm[cur_slice]]
        self._outputs.shape[0].value = len(arr)
        if len(arr) < self._batch_size:
            self._finished = True
        self._outputs['cpu'] = arr
        self._batch_num += 1


class data_entry_element(UserGraph):

    has_back = False

    def __init__(self, data_op, previous_element=None):
        fwd_op = data_op
        self._data_op = data_op
        super().__init__(forward_operation=fwd_op, previous_elements=previous_element)

    def reset(self):
        self._data_op.reset()


class Distro:

    def __init__(self, data, labels, batch_size=64, num_gpus=1, shuffle=True, test_split=None):
        super().__init__()
        assert len(data) == len(labels)
        self._data = data
        self._labels = labels
        self._batch_size = batch_size
        self._num_gpus = num_gpus
        if test_split is not None:
            assert isinstance(test_split, float) and test_split > 0. and test_split <= 1.
            split = np.random.permutation(len(data))
            split_point = int(np.floor(len(data) * test_split))
            train_split, valid_split = split[:split_point], split[split_point:]
            data_t, data_v = data[train_split], data[valid_split]
            labels_t, labels_v = labels[train_split], labels[valid_split]
            data = [data_t, data_v]
            labels = [labels_t, labels_v]
        elif not isinstance(data, list):
            data = [data]
            labels = [labels]

        if rm.is_cuda_active():
            data_op = dispatch(data, num_gpus=num_gpus, batch_size=batch_size, shuffle=shuffle)
            lbls_op = dispatch(labels, num_gpus=num_gpus, batch_size=batch_size, shuffle=shuffle)
        else:
            data_op = dispatch_cpu(data, num_gpus=num_gpus, batch_size=batch_size, shuffle=shuffle)
            lbls_op = dispatch_cpu(labels, num_gpus=num_gpus,
                                   batch_size=batch_size, shuffle=shuffle)
        data_op.attach(lbls_op)

        self._dt_op = data_op
        self._lb_op = lbls_op
        self._data_graph = data_entry_element(data_op)
        self._label_graph = data_entry_element(lbls_op)

    def forward(self):
        pass

    def get_output_graphs(self):
        self._data_graph.detach()
        self._label_graph.detach()
        return self._data_graph, self._label_graph

    def change_data(self, new_data):
        self._dt_op.value = new_data

    def change_label(self, new_label):
        self._lb_op.value = new_label

    def reset(self):
        self._dt_op.reset()
        self._lb_op.reset()

    def __repr__(self):
        return self._data_graph.__repr__()

    def __len__(self):
        '''
            Returns number of iteration N.
            N = ceil(data_size / batch_size)
        '''
        return np.ceil(len(self._data) / float(len(self._batch_size)))
