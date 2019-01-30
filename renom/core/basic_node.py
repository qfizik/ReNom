# -*- coding: utf-8 -*-
from __future__ import print_function, division
import collections
import weakref
import numpy as np
from numbers import Number
from renom import precision, __version__
import renom.debug_graph
import renom.cuda
if renom.cuda.has_cuda():
    from renom.cuda.base import cuda_base
    from renom.cuda.gpuvalue.gpuvalue import GPUValue


class GraphAttrs(object):

    def __init__(self):
        object.__setattr__(self, 'v__attrs', {})

    def clear(self):
        self.v__attrs.clear()

    def get_names(self):
        return self.v__attrs.keys()

    def get_attrs(self):
        return self.v__attrs.values()

    def __setattr__(self, name, value):
        self.v__attrs[name] = value

    def __getattr__(self, name):
        try:
            return self.v__attrs[name]
        except KeyError:
            raise AttributeError('%r has no attribute %r' % (self, name))

    def get(self, key, default=None):
        return self.v__attrs.get(key, default)


class Node(np.ndarray):
    '''This is the base class of all of the auto-differentiation
    compatible array. Using this array class for calculating,
    the calculation history(computational graph) will be built and the gradient of 
    any node class object on the computational gradient can be calculated. 

    Node object can be initialized giving numpy array. If the data type of given array is
    not float32 or float64, Node object automatically casts it to float32 or float64
    according to the ``renom.precision`` setting. By default, renom.precision is set to
    float32. You can use float64 precision by setting an environment variable 
    RENOM_PRECISION to be 64. For example, following shell script set the environment variable.

    .. code-block:: shell

        export RENOM_PRECISION=64


    As the Node class is a base class, user might not deal with object of Node class.
    ReNom provides other auto-differentiation compatible array class called Variable.
    The Variable class has more utility interfaces such as ``weight_decay``, ``auto_update``.
    This helps users to manage the arrays and earned gradients.
    For more information, please refer to the reference of :class:`Variable`.

    Following example is basic case of getting a gradient of an array.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> 
        >>> v1 = rm.Variable(np.array([1, 2]))
        >>> v2 = rm.Variable(np.array([3, 4]))
        >>> result = rm.sum(v1 * v2)
        >>> print("Result", result)
        Result 11.0
        >>> 
        >>> grads = result.grad()
        >>> print("Gradient of v1 is", grads.get(v1))
        Gradient of v1 is [3. 4.]
        >>> 
        >>> print("Gradient of v1 is", grads.get(v2))
        Gradient of v1 is [1. 2.]


    If cuda is activated, Node object automatically uses the gpu device for calculation.
    Once array data transferred to gpu device, the array data will be kept on gpu memory.
    For this reason, users need to call ``as_ndarray`` method for transferring the array data
    from gpu memory to cpu memory. The method as_ndarray returns an array casted to ndarray object.
    We recommend to call ``as_ndarray`` any time for checking the calculation result.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> from renom.cuda import set_cuda_active
        >>> 
        >>> set_cuda_active(True)
        >>> x1 = np.random.rand(2, 2)
        >>> x2 = np.random.rand(2, 2)
        >>> node1 = rm.Node(x1)
        >>> node2 = rm.Node(x2)
        >>> 
        >>> # This add operation performed in gpu device.
        >>> result = node1 + node2
        >>> print(result.as_ndarray()) # Transferring the array data. (gpu => cpu)
        [[1.0171516  1.0086492 ]
         [1.4602256  0.77632725]]

    '''

    _gpu = None
    attrs = None
    _model = None
    _auto_update = False
    _no_backward = False
    _args = ()
    SHOWMARK = False
    _node_hook = None

    @classmethod
    def set_hook(cls, hook):
        cls._node_hook = hook

    def __new__(cls, value):
        ret = cls._create_node(value)
        return ret

    @classmethod
    def _run_node_hook(cls, ret):
        if cls._node_hook:
            ret = cls._node_hook.leave_create(cls, ret)
        return ret

    @classmethod
    def _create_node(cls, value):
        if isinstance(value, np.ndarray):
            ret = value.astype(precision).view(cls)
        elif renom.cuda.has_cuda() and isinstance(value, GPUValue):
            ret = super(Node, cls).__new__(
                cls, shape=value.shape, dtype=value.dtype)
            ret._gpu = value

        elif isinstance(value, Number):
            ret = np.array(value, dtype=precision).view(cls)
        else:
            raise ValueError('Invalid Node value: %r' % value)

        assert ret.dtype == precision, (
            'Type miss matched. Required is {}, actual is {}'.format(
                precision().dtype, ret.dtype))

        ret.attrs = GraphAttrs()
        if renom.debug_graph.GET_ACTIVE_NODE() is not None:
            renom.debug_graph.SET_NODE_DICT(id(ret), ret)

        ret = cls._run_node_hook(ret)

        return ret

    @classmethod
    def calc_value(cls, *args, **kwargs):
        if renom.cuda.is_cuda_active():
            value = cls._oper_gpu(*args, **kwargs)
        else:
            value = cls._oper_cpu(*args, **kwargs)
        return value

    def __init__(self, *args, **kwargs):
        self.setflags(write=False)
        self._args = []
        q = collections.deque([args])
        while q:
            a = q.pop()
            if isinstance(a, Node):
                self._args.append(a)
            elif isinstance(a, list) or isinstance(a, tuple):
                q.extend(a)
            elif isinstance(a, dict):
                q.extend(a.values())
        self._args.extend(a for a in kwargs.values() if isinstance(a, Node))

        self._reduce_graph()
        return

    @property
    def auto_update(self):
        '''If this is True, gradient related to this object will be calculated.
        '''
        if self._auto_update:
            if self._model:
                if not self._model.auto_update:
                    return False
        return self._auto_update

    @auto_update.setter
    def auto_update(self, value):
        raise Exception()

    @property
    def prevent_update(self):
        if self._model:
            if self._model._prevent_update:
                return True
        return False

    @prevent_update.setter
    def prevent_update(self, value):
        raise Exception()

    @property
    def device_id(self):
        if self._gpu:
            return self._gpu.device_id

        if self._model:
            return self._model._device_id

        return 0

    def set_model(self, model):
        self._model = model

    def get_gpu(self):
        '''This function transfers array data to gpu device and
        returns it as a GPUValue object.
        For imformation of GPUValue class please refer :class:`GPUValue`.

        Example:
            >>> import numpy as np
            >>> import renom as rm
            >>> from renom.cuda import set_cuda_active
            >>> 
            >>> v = rm.Variable(np.array([1, 2]))
            >>> print(v.get_gpu()) # This raises error without cuda.
            ValueError: Cuda is not active.
                Use renom.cuda.set_cuda_active() to activate.
            >>>
            >>> set_cuda_active()
            >>> array = v.get_gpu()
            >>> print(array)
            array([1., 2.], dtype=float32)
            >>> print(type(array))
            <class 'renom.cuda.gpuvalue.gpuvalue.GPUValue'>

        Returns:
            (GPUValue): Matrix transferred to gpu device.
        '''
        if not self._gpu:
            self._gpu = GPUValue(self)
        return self._gpu

    def set_gpu(self, gpu):
        self.release_gpu()
        self._gpu = gpu

    def to_cpu(self):
        '''Transfer the data from GPU device to CPU.'''
        if self._gpu:
            self._gpu.to_cpu(self)

    def to_gpu(self):
        '''Transfer the data on CPU to GPU device.
        This method only available if cuda is activated otherwise this raises `ValueError`.
        '''
        if self._gpu:
            self._gpu.to_gpu(self)
        else:
            self._gpu = GPUValue(self)

    def copy(self):
        '''Returns a copy of itself.
        If cuda is not activated, this method returns ndarray.

        Returns:
            (Node, ndarray): Copy of node object.
        '''
        if self._gpu:
            return self.__class__(self._gpu.copy())
        else:
            return np.ndarray.copy(self)

    def copy_from(self, other):
        assert self.shape == other.shape
        assert self.dtype == other.dtype
        if self._gpu:
            if other._gpu:
                self._gpu.copy_from(other._gpu)
                return

        if hasattr(self, 'setflags'):
            writable = self.flags.writeable
            self.setflags(write=True)

        try:
            self[...] = other
        finally:
            if hasattr(self, 'setflags'):
                self.setflags(write=writable)

    def as_ndarray(self):
        '''This method returns array casted to numpy ndarray object.

        Returns:
            (ndarray): Returns an array as a ndarray object.

        Example:
            >>> import numpy as np
            >>> import renom as rm
            >>> v = rm.Variable(np.array([1, 2]))
            >>> array = v.as_ndarray()
            >>> print("Data:", array)
            Data: [1. 2.]
            >>> print("Object type:", type(array))
            Object type: <class 'numpy.ndarray'>

        '''
        self.to_cpu()
        if self._gpu:
            return self._gpu.new_array()
        if isinstance(self, Number):
            return np.array(self, dtype=precision)
        else:
            if not self.flags['C_CONTIGUOUS']:
                self = np.ascontiguousarray(self)
            ret = np.ndarray(shape=self.shape, dtype=self.dtype, buffer=self)
            ret.setflags(write=True)
            return np.array(ret)

    def release_gpu(self):
        '''This method releases array data on GPU pointed by this object.'''
        if self._gpu:
            self._gpu = None

    def _update_diff(self, context, dy, **kwargs):
        ready = context.add(self, dy)
        if ready:
            diff = context.get(self)
            self.backward(context, diff, **kwargs)

    def _get_graph(self):
        if self.attrs:
            return self.attrs.get_attrs()
        return []

    def _has_autoupdate(self):
        '''Check if the graph to witch this node belongs need to update.'''

        for v in self._get_graph():
            if isinstance(v, Node):
                if v.auto_update:
                    return True

                if any((o is not None) for o in v._get_graph()):
                    return True

    def _reduce_graph(self):
        if self.attrs:
            if not self._has_autoupdate():
                self._no_backward = True
                self.attrs.clear()
                self._args = []
        return False

    def detach_graph(self):
        '''This method destroys computational graph.
        As following example, once this method is called, 
        gradients can't be calculated because computational is removed.
        This example raises an error that mentions Node object was not found on
        the computational graph.

        Example:
            >>> import numpy as np
            >>> import renom as rm
            >>> 
            >>> v1 = rm.Variable(np.array([1, 2]))
            >>> v2 = rm.Variable(np.array([3, 4]))
            >>> result = rm.sum(v1 * v2)
            >>> print("Result", result)
            >>> result.detach_graph()
            >>> grads = result.grad()
            >>> print("Gradient of v1 is", grads.get(v1))
            Exception: Node not found.
                Ensure that _update_diff was properly called on the node first.
        '''
        for v in self._get_graph():
            if isinstance(v, Node):
                v.detach_graph()
        if self.attrs:
            self.attrs.clear()

        self._args = []

    def backward(self, context, dy, **kwargs):
        if self._no_backward:
            return

        if renom.cuda.is_cuda_active():
            if self._gpu:
                with cuda_base.use_device(self._gpu.device_id):
                    return self._backward_gpu(context, dy, **kwargs)
            else:
                return self._backward_gpu(context, dy, **kwargs)
        else:
            return self._backward_cpu(context, dy, **kwargs)

    def __neg__(self):
        assert False

    def __pos__(self):
        return renom.core.Pos(self)

    def __abs__(self):
        assert False

    def __invert__(self):
        assert False

    def __add__(self, other):
        '''This method is defined in basic_ops.py'''
        assert False

    def __radd__(self, other):
        '''This method is defined in basic_ops.py'''
        assert False

    def __iadd__(self, other):
        '''This method is defined in basic_ops.py'''
        assert False

    def __sub__(self, other):
        '''This method is defined in basic_ops.py'''
        assert False

    def __rsub__(self, other):
        assert False

    def __isub__(self, other):
        assert False

    def __mul__(self, other):
        '''This method is defined in basic_ops.py'''
        assert False

    def __rmul__(self, other):
        assert False

    def __imul__(self, other):
        assert False

    def __div__(self, other):
        assert False

    def __rdiv__(self, other):
        assert False

    def __idiv__(self, other):
        assert False

    def __floordiv__(self, other):
        assert False

    def __rfloordiv__(self, other):
        assert False

    def __ifloordiv__(self, other):
        assert False

    def __truediv__(self, other):
        assert False

    def __rtruediv__(self, other):
        assert False

    def __itruediv__(self, other):
        assert False

    def __mod__(self, other):
        assert False

    def __rmod__(self, other):
        assert False

    def __imod__(self, other):
        assert False

    def __divmod__(self, other):
        assert False

    def __rdivmod__(self, other):
        assert False

    def __pow__(self, other):
        assert False

    def __rpow__(self, other):
        assert False

    def __ipow__(self, other):
        assert False

    def __lshift__(self, other):
        assert False

    def __rlshift__(self, other):
        assert False

    def __ilshift__(self, other):
        assert False

    def __rshift__(self, other):
        assert False

    def __rrshift__(self, other):
        assert False

    def __irshift__(self, other):
        assert False

    def __and__(self, other):
        assert False

    def __rand__(self, other):
        assert False

    def __iand__(self, other):
        assert False

    def __xor__(self, other):
        assert False

    def __rxor__(self, other):
        assert False

    def __ixor__(self, other):
        assert False

    def __or__(self, other):
        assert False

    def __ror__(self, other):
        assert False

    def __ior__(self, other):
        assert False

    def __getitem__(self, index):
        '''This method is defined in basic_ops.py'''
        assert False

    def __setitem__(self, index, value):
        if self._gpu is not None:
            self._gpu[index] = value
        else:
            np.ndarray.__setitem__(self, index, value)

    def __getslice__(self, i, j):
        assert False

    def __lt__(self, other):
        self.to_cpu()
        if hasattr(other, 'to_cpu'):
            other.to_cpu()
        return np.ndarray.__lt__(self, other)

    def __le__(self, other):
        self.to_cpu()
        if hasattr(other, 'to_cpu'):
            other.to_cpu()
        return np.ndarray.__le__(self, other)

    def __eq__(self, other):
        self.to_cpu()
        if hasattr(other, 'to_cpu'):
            other.to_cpu()
        return np.ndarray.__eq__(self, other)

    def __ne__(self, other):
        self.to_cpu()
        if hasattr(other, 'to_cpu'):
            other.to_cpu()
        return np.ndarray.__ne__(self, other)

    def __ge__(self, other):
        self.to_cpu()
        if hasattr(other, 'to_cpu'):
            other.to_cpu()
        return np.ndarray.__ge__(self, other)

    def __gt__(self, other):
        self.to_cpu()
        if hasattr(other, 'to_cpu'):
            other.to_cpu()
        return np.ndarray.__gt__(self, other)

    def __not__(self):
        self.to_cpu()
        return np.ndarray.__not__(self)

    def __str__(self):
        self.to_cpu()
        return np.ndarray.__str__(self.as_ndarray())

    def __repr__(self):
        self.to_cpu()
        return np.ndarray.__repr__(self)

    def __float__(self):
        self.to_cpu()
        return np.ndarray.__float__(self)

    def __int__(self):
        self.to_cpu()
        return np.ndarray.__int__(self)

    def __complex__(self):
        self.to_cpu()
        return np.ndarray.__complex__(self)

    def __bool__(self):
        self.to_cpu()
        return np.ndarray.__bool__(self)

    def __index__(self):
        self.to_cpu()
        return np.ndarray.__index__(self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # move gpu values of input arrays to cpu
        new_inputs = []
        for item in inputs:
            if isinstance(item, Node):
                item.to_cpu()
                item.release_gpu()
                new_inputs.append(item.view(np.ndarray))
            else:
                new_inputs.append(item)

        # move gpu values of output arrays to cpu
        outs = kwargs.get('out', None)
        if isinstance(outs, tuple):
            new_outs = []
            for item in outs:
                if isinstance(item, Node):
                    item.to_cpu()
                    item.release_gpu()
                    new_outs.append(item.view(np.ndarray))
                else:
                    new_outs.append(item)

            kwargs['out'] = tuple(new_outs)

        elif outs is not None:
            kwargs['out'] = outs.view(np.ndarray)
            outs.to_cpu()
            outs.release_gpu()

        ret = getattr(ufunc, method)(*new_inputs, **kwargs)
        return ret

    def reshape(self, *shape):
        '''This method is defined in basic_ops.py'''
        assert False


class Variable(Node):
    '''Variable class. The gradient of this object will be calculated.
    Variable object is created from ndarray object or Number object.

    Args:
        value (Variable, ndarray, Number): Input array.
        auto_update (bool): Auto update flag.
        weight_decay (float): Weight decay rate

    Weight decay allows the user to choose if weight decay is to be used in any
    of their variables.
    If weight decay is not defined in the Variable (I.e. defaults to None),
    then no weight decay is performed.

    For convenience, one can define a variable with a weight decay of 0 and provide
    the weight decay argument when building the gradients to default all weights to the
    same λ for weight decay.

    Individually assigned weight decay takes precedence over this default value,
    allowing users to customize the weight decay in the network.

    In summary, weight decay updates according to the following table.

    +-----------+-----------+--------------+
    | Variable  |   Grad    |   Result     |
    +===========+===========+==============+
    | None      |   <Any>   |   No Update  |
    +-----------+-----------+--------------+
    | 0.3       |   <Any>   |   0.3        |
    +-----------+-----------+--------------+
    | 0         |   None/0  |   No Update  |
    +-----------+-----------+--------------+
    | 0         |   0.3     |   0.3        |
    +-----------+-----------+--------------+

    '''

    weight_decay = None
    '''Weight decay coefficient.'''

    def __new__(cls, value, auto_update=True, weight_decay=None):
        ret = super(Variable, cls).__new__(cls, value)
        ret._auto_update = auto_update
        ret.weight_decay = weight_decay
        return ret

    def __init__(self, value, auto_update=True, weight_decay=None):
        pass

    def backward(self, context, dy, **kwargs):
        pass
