import renom as rm
from renom.graph.core import operation, UserGraph, GraphMultiStorage, GraphFactory
import numpy as np


class lrn_forward(operation):

    name = 'LRN (F)'

    def __init__(self, n=5, k=2, a=1e-4, b=0.75):
        self._n = n
        self._k = k
        self._a = a
        self._b = b

    def setup(self, inputs):

        inputs = inputs[0]['y']
        input_shape = inputs.shape
        self._inputs = inputs

        out_shape = input_shape

        self.gpus = inputs.gpus
        outs = GraphMultiStorage(shape=out_shape, gpus=self.gpus)
        self._outputs = outs
        self._vars = {'y': outs}
        if rm.is_cuda_active():
            lrn_desc = rm.cuda.LRNDescriptor(self._n, self._a, self._b, self._k)
            self._desc = lrn_desc

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cuLocalResponseNormalizationForward(
                handle, self._desc, self._inputs[gpu], self._outputs[gpu])


class lrn_forward_cpu(lrn_forward):

    def perform(self):
        x = self._inputs['cpu']
        xs = np.square(x).view(np.ndarray)
        tmp = xs.copy()
        for i in range(1, self._n // 2 + 1):
            tmp[:, i:, :, :] += xs[:, :-i, :, :]
            tmp[:, :-i, :, :] += xs[:, i:, :, :]
        unit_scale = self._k + self._a * tmp
        self._unit_scale = unit_scale
        scale = unit_scale ** -self._b
        self._scale = scale
        value = x * scale
        self._outputs['cpu'] = value


class lrn_backward(operation):

    name = 'LRN (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):

        inputs = inputs[0]['y']
        self._inputs = inputs
        out_shape = self._fwd_op._inputs.shape
        self._fwd_in = self._fwd_op._inputs
        self._fwd_out = self._fwd_op._outputs
        self.gpus = inputs.gpus
        outs = GraphMultiStorage(shape=out_shape, gpus=self.gpus)
        self._outputs = outs
        self._vars = {'y': outs, id(self._fwd_in): outs}
        if rm.is_cuda_active():
            self._desc = self._fwd_op._desc

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cuLocalResponseNormalizationBackward(
                handle, self._desc, self._fwd_in[gpu], self._fwd_out[gpu], self._outputs[gpu], self._inputs[gpu])


class lrn_backward_cpu(lrn_backward):

    def perform(self):
        dy = self._inputs['cpu']
        y = self._fwd_op._outputs['cpu']
        x = self._fwd_op._inputs['cpu']
        tmp1 = (y * dy / self._fwd_op._unit_scale).view(np.ndarray)
        tmp2 = tmp1.copy()
        for i in range(1, self._fwd_op._n // 2 + 1):
            tmp2[:, i:, :, :] += tmp1[:, :-i, :, :]
            tmp2[:, :-i, :, :] += tmp1[:, i:, :, :]
        value = dy * self._fwd_op._scale - 2 * self._fwd_op._a * self._fwd_op._b * x * tmp2
        self._outputs['cpu'] = value


class LrnElement(UserGraph):

    def __init__(self, n=5, k=2, a=1e-4, b=0.75, previous_element=None):
        self._n = n
        self._k = k
        self._a = a
        self._b = b
        fwd_op = lrn_forward(self._n, self._k, self._a, self._b) if rm.is_cuda_active(
        ) else lrn_forward_cpu(self._n, self._k, self._a, self._b)
        bwd_ops = [lrn_backward(fwd_op) if rm.is_cuda_active() else lrn_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_element)


class Lrn(GraphFactory):
    '''Local response normalization function [lrn]_ .

    .. math::
        y_{c_{out},j,i}= \\frac{x_{c_{in},i,j}}{(k + a{\sum_{c=max(0, i-n/2)}^{min(N-1, i+n/2)} (x_{c,i,j})})^b}

    :math:`x_{c,i,j}` represents the c th conv filter’s output at the position of (i, j) in the feature map.
    :math:`y_{c_{out},j,i}` represents the output of local response normalization.
    :math:`N` is the number of the feature maps.
    :math:`n` is the adjacent feature map number.
    The default argument values are recommended.

    Args:
        n (int): Number of images used to be normalized.
        k (int): Constant.
        a (float): Scale factor.
        b (float): Exponential factor.

    Example:
        In [1]: import numpy as np
        In [2]: import renom as rm
        In [3]: x = np.random.rand(3, 3, 32, 32)
        In [4]: layer = rm.graph.LrnGraphElement()
        In [5]: z=layer(x).as_ndarray()
        In [6]: z.shape
        Out[6]:
        (3, 3, 32, 32)

    .. [lrn] Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton.
        ImageNet Classification with Deep Convolutional Neural Networks. NIPS, 2012.

    '''

    def __init__(self, n=5, k=2, a=1e-4, b=0.75):
        super().__init__()
        self._n = n
        self._k = k
        self._a = a
        self._b = b

    def connect(self, other):
        ret = LrnElement(self._n, self._k, self._a, self._b, previous_element=[other])
        return ret