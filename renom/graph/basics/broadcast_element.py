import renom as rm
from renom.graph.core import operation, GraphMultiStorage, operational_element, UserGraph
import numpy as np


class broadcast_forward(operation):

    name = 'Broadcast (F)'

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
        self._c = GraphMultiStorage(shape=a.shape, gpus=self.gpus)
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
        self._outputs = inputs[0]['y']
        self._vars = {'y': self._outputs, 'dy': self._outputs,
                      id(self._fwd_op.get_key(self._key)): self._outputs}

    def perform(self):
        pass


class AddElement(UserGraph):

    _name = 'Add Element'

    def __init__(self, previous_elements=None):

        fwd_op = add_forward() if rm.is_cuda_active() else add_forward_cpu()
        bwd_ops = [add_backward(fwd_op, 'a'), add_backward(fwd_op, 'b')]
        super().__init__(fwd_op, bwd_ops, previous_elements)


def _add(self, other):
    ret = AddElement([self, other])
    return ret


UserGraph.__add__ = _add
