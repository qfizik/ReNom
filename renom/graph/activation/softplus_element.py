import renom as rm
from renom.graph.core import UserGraph, operation, GraphFactory, graph_variable, GraphMultiStorage
import numpy as np


class softplus_forward(operation):

    name = 'SoftPlus (F)'

    def setup(self, inputs):
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus
        in_shape = inputs.shape
        outs = GraphMultiStorage(shape=in_shape, gpus=gpus)
        self._inputs = inputs
        self._outputs = outs
        self._vars = {'y': outs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cusoftplus_forward(self._inputs[gpu], self._outputs[gpu])


class softplus_forward_cpu(softplus_forward):

    def perform(self):
        x = self._inputs['cpu']
        ret = np.log(1 + np.exp(x))
        self._outputs['cpu'] = ret


class softplus_backward(operation):

    name = 'SoftPlus (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus
        in_shape = inputs.shape
        outs = GraphMultiStorage(shape=in_shape, gpus=gpus)
        self._inputs = inputs
        self._outputs = outs
        self._fwd_out = self._fwd_op._outputs
        self._vars = {'y': outs, id(self._fwd_op._inputs): outs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cusoftplus_backward(self._fwd_out[gpu], self._inputs[gpu], self._outputs[gpu])


class softplus_backward_cpu(softplus_backward):

    def perform(self):
        x = self._fwd_op._inputs['cpu']
        dx = 1 / (1 + np.exp(-x))
        dy = self._inputs['cpu']
        ret = dx * dy
        self._outputs['cpu'] = ret


class SoftplusElement(UserGraph):

    def __init__(self, previous_elements=None):
        fwd_op = softplus_forward() if rm.is_cuda_active() else softplus_forward_cpu()
        bwd_ops = [softplus_backward(fwd_op) if rm.is_cuda_active()
                   else softplus_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


class Softplus(GraphFactory):
    '''A factory class of elu activation function element.

    .. math::

        f(x) = log(1 + exp(x))

    Args:
        x (ndarray, Node): Input numpy array or Node instance.

    Example:
        >>> import numpy as np
        >>> import renom.graph as rmg
        >>>
        >>> x = np.array([1., -1.])
        >>>
        >>> layer = rmg.Softplus()
        >>> layer(x)
        Softplus (F):
        [1.31326163, 0.31326169]
        >>>
        >>> rmg.softplus(x)
        Softplus (F):
        [1.31326163, 0.31326169]

    '''

    def connect(self, other):
        ret = SoftplusElement(previous_elements=other)
        return ret


def softplus(x):
    return SoftplusElement(previous_elements=[x])
