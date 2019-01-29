import numpy as np
import renom as rm
from renom.graph.core import operation, GraphMultiStorage, operational_element, UserGraph, GraphFactory
from renom.core import broad_cast, cu_broad_cast


class add_forward(operation):

    name = 'Add (F)'

    def __init__(self):
        self._a = None
        self._b = None

    def setup(self, inputs):
        a = inputs[0]['y']
        b = inputs[1]['y']
        assert len(a) == len(b)
        self.gpus = a.gpus
        self._a = a
        self._b = b
        output_shape = (np.zeros(a.shape) + np.zeros(b.shape)).shape
        self._c = GraphMultiStorage(shape=output_shape, gpus=self.gpus)
        self._vars = {'a': a, 'b': b, 'y': self._c}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cuadd(self._a[gpu], self._b[gpu], self._c[gpu], handle)


class add_forward_cpu(add_forward):

    def perform(self):
        a = self._a['cpu']
        b = self._b['cpu']
        self._c['cpu'] = a + b


class add_backward(operation):

    name = 'Add (B)'

    def __init__(self, associated_forward, key):
        self._fwd_op = associated_forward
        self._key = key

    def setup(self, inputs):
        self._inputs = inputs[0]['dy']
        key = self._key
        a = self._fwd_op.get_key(key)
        gpus = a.gpus
        output_shape = a.shape
        outputs = GraphMultiStorage(shape=output_shape, gpus=gpus, initializer=None)

        self._a = a
        self.gpus = gpus
        self._vars = {'y': outputs, 'dy': outputs, id(a): outputs}
        self._outputs = outputs

    def perform(self):
        for i, (gpu, handle) in enumerate(rm.cuda.RenomHandlers(self.gpus)):
            a = self._a[gpu]
            dy = self._inputs[gpu]
            if a.shape != dy.shape:
                dy = cu_broad_cast(a, dy)
            self._outputs[gpu] = dy


class add_backward_cpu(add_backward):

    def perform(self):
        a = self._a['cpu']
        dy = self._inputs['cpu']
        if a.shape == dy.shape:
            self._outputs['cpu'] = dy
        else:
            self._outputs['cpu'] = broad_cast(a, dy)


class AddElement(UserGraph):

    _name = 'Add Element'

    def __init__(self, previous_elements=None):

        fwd_op = add_forward() if rm.is_cuda_active() else add_forward_cpu()
        bwd_ops = [add_backward(fwd_op, 'a') if rm.is_cuda_active() else add_backward_cpu(fwd_op, 'a'),
                   add_backward(fwd_op, 'b') if rm.is_cuda_active() else add_backward_cpu(fwd_op, 'b')]
        super().__init__(fwd_op, bwd_ops, previous_elements)


class AddGraphElement(GraphFactory):

    def connect(self, lhs, rhs):
        return AddElement([lhs, rhs])


def _add(self, other):
    ret = AddElement([self, other])
    return ret


UserGraph.__add__ = _add
UserGraph.__iadd__ = _add
UserGraph.__radd__ = _add
