import renom as rm
from renom.graph.core import UserGraph, operation, GraphFactory, graph_variable, GraphMultiStorage
import numpy as np


class DropoutGraphElement(GraphFactory):
    """Applies Dropout [dropout]_ to the input.

    Dropout function randomly selects a fraction (specified by dropout_ratio) of
    the data sets them to zero.
    Remaining data will be rescaled by ``1/(1 - dropout_ratio)``.

    Args:
        dropout_ratio (float): Dropout ratio.

    Example:

    .. code-block:: python

        In [1]: import numpy as np
        In [2]: import renom as rm
        In [3]: x = np.random.rand(3,2)
        In [4]: x
        Out[4]:
        array([[ 0.92146051,  0.09946255],
               [ 0.05895275,  0.78195323],
               [ 0.98867317,  0.03215612]])
        In [5]: layer = rm.graph.DropoutGraphElement(0.8)
        In [6]: z = layer(x).as_ndarray()
        In [7]: z
        Out[7]:
        array([[ 0.        ,  0.        ],
               [ 0.11790549,  0.        ],
               [ 1.97734635,  0.        ]])



    """

    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self._dr = dropout_rate

    def connect(self, other):
        ret = DropoutElement(self._dr, previous_elements=other)
        return ret


class dropout_forward(operation):

    name = 'Dropout (F)'
    roles = ['inference']

    def __init__(self, dropout_rate=0.5):
        self._dropout_rate = dropout_rate
        self._inference = False

    def setup(self, inputs):
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus
        mask = GraphMultiStorage(shape=inputs.shape, gpus=gpus)
        outs = GraphMultiStorage(shape=inputs.shape, gpus=gpus)
        self._vars = {'y': outs}
        self._inputs = inputs
        self._outputs = outs
        self._mask = mask

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            if self._inference:
                rm.cuda.cumul(self._inputs[gpu], 1, self._outputs[gpu], handle)
            else:
                rm.cuda.curand_generator().rand_bernoulli(self._mask[gpu], 1 - self._dropout_rate)
                rm.cuda.cudiv(self._mask[gpu], self._dropout_rate, self._mask[gpu], handle)
                rm.cuda.cumul(self._mask[gpu], self._inputs[gpu], self._outputs[gpu], handle)


class dropout_forward_cpu(dropout_forward):

    def perform(self):
        if self._inference:
            x = self._inputs['cpu']
            self._outputs['cpu'] = x
        else:
            x = self._inputs['cpu']
            dropout_ratio = 1 - self._dropout_rate
            mask = np.array(np.random.rand(*x.shape) < dropout_ratio,
                            dtype=rm.precision) / dropout_ratio
            ret = x * mask
            self._mask['cpu'] = mask
            self._outputs['cpu'] = ret


class dropout_backward(operation):

    name = 'Dropout (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus
        outs = GraphMultiStorage(shape=inputs.shape, gpus=gpus)
        self._vars = {'y': outs, id(self._fwd_op._inputs): outs}
        self._fwd_mask = self._fwd_op._mask
        self._outputs = outs
        self._inputs = inputs

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cumul(self._inputs[gpu], self._fwd_mask[gpu], self._outputs[gpu], handle)


class dropout_backward_cpu(dropout_backward):

    def perform(self):

        dy = self._inputs['cpu']
        mask = self._fwd_mask['cpu']
        ret = dy * mask
        self._outputs['cpu'] = ret


class DropoutElement(UserGraph):

    def __init__(self, dropout_rate=0.5, previous_elements=None):
        self.dropout_ratio = dropout_rate
        fwd_op = dropout_forward() if rm.is_cuda_active() else dropout_forward_cpu()
        bwd_ops = [dropout_backward(fwd_op) if rm.is_cuda_active()
                   else dropout_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)
