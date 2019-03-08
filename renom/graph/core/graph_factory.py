import renom as rm
import abc
import numpy as np
from .user_graph import UserGraph
from .operation import operation
from .operational_element import unidirectional_element
from .graph_storage import GraphMultiStorage
import contextlib as cl
import functools
import h5py


def recursive_setting(func):
    @functools.wraps(func)
    def ret_func(self, *args, **kwargs):
        func(self, *args, **kwargs)
        for elem in self.__dict__.values():
            if isinstance(elem, GraphFactory):
                ret_func(elem, *args, **kwargs)
    return ret_func


class GraphFactory(abc.ABC):

    '''
        A utility class designed to remove the user from the states of the UserGraph implementing
        classes. The GraphFactory provides two services, allowing the user to rebuild the graph
        without worrying about states of connection between UserGraph elements and removing the
        responsibility of maintaining weights and biases from the UserGraph class.

        A class implementing the abstract GraphFactory class is required only to implement the
        connect method which should return a prepared UserGraph. If an implementing class inherits
        __init__, the overrided __init__ should also be called.

        If there is a need for weights or biases, etc, then these variables should be stored in the
        params dictionary which is instantiated for each GraphFactory object. This allows GraphFactory
        to detach these variables when reconstructing the graph as well as save/load the variables
        later using the save/load methods.
    '''

    def __init__(self):
        '''Generic UserGraph initializer.

            Initializes the parameter field as an empty dictionary and
            sets the previous output to None.

        '''
        self.params = dict()
        self._prev = None
        self._inference = None
        self._make_update_graphs = True

    @abc.abstractmethod
    def connect(self, other):
        '''This method connects the UserGraph to another graph.

            If the previous output is None, the GraphFactory will \
            attempt to detach the previous UserGraph output to produce \
            a disconnected graph.
        '''
        pass

    def __call__(self, *other):
        self_tag = id(self)
        with rm.graph.core._with_operational_tag(self_tag):
            ret = self.connect(*other)
        ret._fwd.replace_tags(self_tag, id(ret))
        for bwd in ret._bwd_graphs:
            bwd.replace_tags(self_tag, id(ret))
        if not self._make_update_graphs:
            for op_num, upd_g in ret._update_graphs:
                upd_g.detach()
        if self._inference is not None:
            ret._fwd._op._inference = True
        self._prev = ret
        return ret

    @recursive_setting
    def _set_make_updates(self, make_updates=True):
        self._make_update_graphs = make_updates

    @cl.contextmanager
    def no_updates(self):
        self._set_make_updates(False)
        yield
        self._set_make_updates(True)

    @cl.contextmanager
    def inference(self):
        self.set_inference(False)
        yield
        self.set_inference(True)

    @recursive_setting
    def set_inference(self, infer=True):
        self._inference = infer

    def _get_model_children(self):
        for k, v in self.__dict__.items():
            if isinstance(v, GraphFactory):
                yield k, v

    def __getitem__(self, name):
        return self.params[name]

    def __setitem__(self, name, value):
        self.params[name] = value

    def _get_values(self, values):
        if self.params:
            for k in self.params.keys():
                values[1][k] = self.params[k]

        serialized = getattr(self, "SERIALIZED", ())
        for name in serialized:
            if hasattr(self, name):
                values[2][name] = getattr(self, name)

        for k, v in self._get_model_children():
            childvalues = ({}, {}, {})
            v._get_values(childvalues)
            values[0][k] = childvalues

    def _values(self):
        ret = ({}, {}, {})
        self._get_values(ret)
        return ret

    def _flatten_values(self):
        values = self._values()
        value_list = []

        def flatten(names, values):
            value_list.append((names, values[1], values[2]))

            for name, child_values in values[0].items():
                flatten(names + (name,), child_values)

        flatten(('root',), values)
        return value_list

    def load_v2_params(self, other_params, gpus=None):
        self_params = self.params
        for key in other_params:
            if key in self_params:
                other_val = other_params[key].as_ndarray()
                self_params[key].set_value(other_val)

    def save(self, filename):
        """Save model attributes.
        For save attributes, please register attributes to the dictionary
        which is named as 'SERIALIZED'.

        Following example shows how to do it.

        Example:
            >>> import renom as rm
            >>> import numpy as np
            >>>
            >>> class MyModel(rm.Model):
            ...     SERIALIZED = ('_moving_avg', ) # Put any attributes for saving.
            ...     def __init__(self):
            ...         super(MyModel, self).__init__()
            ...         self._l1 = rm.Dense(2)
            ...         self._l2 = rm.Dense(1)
            ...         self._moving_avg = 0
            ...     def forward(self, x):
            ...         h = self._l1(x)
            ...         h = rm.relu(h)
            ...         h = self._l2(h)
            ...         self._moving_avg = np.float32(self._moving_avg*0.5 + rm.sum(h)*0.5)
            ...         return h
            ...
            >>> model = MyModel()
            >>> model(np.random.rand(12, 4))
            >>> print(model._moving_avg)
            1.95637
            >>> model.save("test.h5") # Save
            >>> model = MyModel() # Reset model object.
            >>> model.load("test.h5") # Load
            >>> print(model._moving_avg)
            1.95637

        Args:
            filename (str): File name to save model.

        """

        value_list = self._flatten_values()
        with h5py.File(filename, 'w') as f:
            values_grp = f.create_group('values')
            types_grp = f.create_group('types')

            for names, params, attrs in value_list:
                g = values_grp.create_group('.'.join(names))
                t = types_grp.create_group('.'.join(names))

                for propname, propvalue in params.items():
                    propvalue = propvalue.as_ndarray()
                    g[propname] = propvalue

                    t[propname] = 'renom.Variable'
                    t[propname + '._auto_update'] = True

                for propname, propvalue in attrs.items():
                    if isinstance(propvalue, GPUValue):
                        g['__dict__.' + propname] = propvalue.new_array()
                    else:
                        g['__dict__.' + propname] = propvalue

    def load(self, filename, devices=None):
        """Load saved weights to model.

        Args:
            filename (str): File name of saved model.

        Example:
            >>> model = rm.Dense(2)
            >>> model.load("model.hd5")
        """
        if devices is None:
            if rm.is_cuda_active():
                devices = 1
            else:
                devices = 'cpu'

        if isinstance(devices, int):
            devices = [d for d in range(devices)]

        f = h5py.File(filename, 'r')
        values = f['values']
        #types = f['types']

        names = sorted(values.keys())

        def get_attr(root, names):
            names = names.split('.')[1:]
            ret = root
            for name in names:
                ret = getattr(ret, name)
            return ret

        target = self
        for name in names:
            target = get_attr(self, name)

            values_grp = values[name]
            #types_grp = types[name]

            for k, v in values_grp.items():
                v = v.value

                if k.startswith('__dict__.'):
                    obj = target
                    name = k.split(".", 1)[1]
                else:
                    obj = target.params
                    name = k

                tmp = obj[name]
                if isinstance(tmp, graph_variable):
                    tmp.set_value(v, gpus=devices)
                else:
                    obj[name] = v

        f.close()


class graph_variable(UserGraph):

    '''
        A simple helper class designed to allow the operations access to 'free' memory.
        The graph_variable provides a single forward operation, variable_input, which
        does nothing but provide a constant output.

        Any operation requiring a modifiable parameter, which cannot be determined
        before the size of the graph is determined, can use the output produced by
        variable_input. This memory is set calling the __init__ method of the
        GraphMultiStorage output. If this memory is already set, it simply reaffirms
        that the same size if request and if not, raises an exception.
    '''

    def set_weight_decay(self, weight_decay):
        self._fwd_op._val.set_weight_decay(weight_decay)

    def allow_update(self, should_allow):
        self._fwd._op._val.set_updatable(should_allow)

    def __init__(self, weight_decay=None, allow_update=True):
        fwd_op = variable_input(weight_decay, allow_update)
        self._fwd_op = fwd_op
        fwd_graph = unidirectional_element(fwd_op, tags=['Forward'])
        bwd_ops = []
        super().__init__(forward_operation=fwd_graph, backward_operations=bwd_ops)

    def set_value(self, arr, gpus=None):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr).reshape(1, 1)
        assert isinstance(arr, np.ndarray)
        v = self._fwd_op.get_key('y')
        v.__init__(shape=arr.shape, gpus=gpus)
        if v.gpus == 'cpu':
            v['cpu'] = arr
        else:
            for gpu in v.gpus:
                v[gpu].to_gpu(arr)


class variable_input(operation):

    name = 'Variable'
    roles = ['variable']

    def __init__(self, weight_decay=None, allow_update=True):
        val = GraphMultiStorage()
        val.set_weight_decay(weight_decay)
        val.set_updatable(allow_update)
        self._val = val
        self._vars = {'y': val}

    def setup(self, inputs):
        pass

    def perform(self):
        pass
