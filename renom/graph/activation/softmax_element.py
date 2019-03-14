import renom as rm
from renom.graph.core import UserGraph, operation, GraphFactory, graph_variable, GraphMultiStorage
import numpy as np


class softmax_forward(operation):

    name = 'SoftMax (F)'

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
            rm.cuda.cuSoftmaxForward(handle, self._inputs[gpu], self._outputs[gpu], mode=1)


class softmax_forward_cpu(softmax_forward):

    def perform(self):
        x = self._inputs['cpu']
        maxes = np.max(x, axis=1, keepdims=True)
        u = np.exp(x - maxes)
        summed = np.sum(u, axis=1, keepdims=True)
        ret = u / (summed + 1e-8)
        self._outputs['cpu'] = ret


class softmax_backward(operation):

    name = 'SoftMax (B)'

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
            rm.cuda.cuSoftmaxBackward(
                handle, self._fwd_out[gpu], self._inputs[gpu], self._outputs[gpu], mode=1)


class softmax_backward_cpu(softmax_backward):

    def perform(self):
        dy = self._inputs['cpu']
        y = self._fwd_out['cpu']
        dx = y * dy
        summed = dx - np.sum(dx, axis=1, keepdims=True)
        ret = ((1.0 - y) * dy * summed) * y
        self._outputs['cpu'] = ret


class SoftmaxElement(UserGraph):

    def __init__(self, previous_elements=None):
        fwd_op = softmax_forward() if rm.is_cuda_active() else softmax_forward_cpu()
        bwd_ops = [softmax_backward(fwd_op) if rm.is_cuda_active()
                   else softmax_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


class Softmax(GraphFactory):
    '''A factory class of softmax activation function element.

    .. math::

        y \in R^{N \\times J},
        y_{nj} = \\frac{exp(x_{nj})}{\\sum_{k}^{J}{exp(x_{nk})}}

    Args:
        alpha (float): Alpha coefficient for Elu.

    Example:
        >>> import numpy as np
        >>> import renom.graph as rmg
        >>>
        >>> x = np.array([-1, 0, 1])
        >>>
        >>> layer = rmg.Elu()
        >>> layer(x)
        Elu (F):
        [-0.00632121  0.          1.        ]
        >>>
        >>> # Create element using function interface.
        >>> rmg.elu(x)
        Elu (F):
        [-0.00632121  0.          1.        ]

    '''

    def connect(self, other):
        ret = SoftmaxElement(previous_elements=other)
        return ret


def softmax(x):
    return SoftmaxElement(previous_elements=[x])
