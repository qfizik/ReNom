import numpy as np
import renom as rm
from renom.graph.core import operation, GraphMultiStorage, operational_element, UserGraph, GraphFactory
from renom.core import broad_cast, cu_broad_cast


class mul_forward(operation):

    name = 'Mul (F)'

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
            rm.cuda.cumul(self._a[gpu], self._b[gpu], self._c[gpu], handle)


class mul_forward_cpu(mul_forward):

    def perform(self):
        a = self._a['cpu']
        b = self._b['cpu']
        self._c['cpu'] = a * b


class mul_backward(operation):

    name = 'Mul (B)'

    def __init__(self, associated_forward, key):
        self._fwd_op = associated_forward
        self._key = key

    def setup(self, inputs):
        self._inputs = inputs[0]['dy']
        key = self._key
        key_value = self._fwd_op.get_key(key)
        gpus = key_value.gpus
        output_shape = key_value.shape
        outputs = GraphMultiStorage(shape=output_shape, gpus=gpus, initializer=None)

        a = self._fwd_op.get_key("a")
        b = self._fwd_op.get_key("b")
        self._a = a if key == "a" else b
        self._b = b if key == "a" else a
        self.gpus = gpus
        self._vars = {'y': outputs, 'dy': outputs, id(key_value): outputs}
        self._outputs = outputs

    def perform(self):
        for i, (gpu, handle) in enumerate(rm.cuda.RenomHandlers(self.gpus)):
            a = self._a[gpu]
            b = self._b[gpu]
            dy = self._inputs[gpu]
            if a.shape != dy.shape:
                dy = cu_broad_cast(a, dy * b)
            else:
                dy = dy * b
            self._outputs[gpu] = dy


class mul_backward_cpu(mul_backward):

    def perform(self):
        a = self._a['cpu']
        b = self._b['cpu']
        dy = self._inputs['cpu']
        if a.shape == dy.shape:
            self._outputs['cpu'] = dy * b
        else:
            self._outputs['cpu'] = broad_cast(a, dy * b)


class MulElement(UserGraph):

    _name = 'Mul Element'

    def __init__(self, previous_elements=None):

        fwd_op = mul_forward() if rm.is_cuda_active() else mul_forward_cpu()
        bwd_ops = [mul_backward(fwd_op, 'b') if rm.is_cuda_active() else mul_backward_cpu(fwd_op, 'b'),
                   mul_backward(fwd_op, 'a') if rm.is_cuda_active() else mul_backward_cpu(fwd_op, 'a')]
        super().__init__(fwd_op, bwd_ops, previous_elements)


class MulGraphElement(GraphFactory):

    def connect(self, lhs, rhs):
        return MulElement([lhs, rhs])


def _mul(self, other):
    ret = MulElement([self, other])
    return ret


UserGraph.__mul__ = _mul
UserGraph.__imul__ = _mul
UserGraph.__rmul__ = _mul
