import renom as rm
from renom.graph.core import UserGraph, operation, GraphFactory, graph_variable, GraphMultiStorage
import numpy as np


class elu_forward(operation):

    name = 'Elu (F)'

    def __init__(self, alpha):
        self._alpha = alpha

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
            rm.cuda.cueru_forward(self._alpha, self._inputs[gpu], self._outputs[gpu])


class elu_forward_cpu(elu_forward):

    def perform(self):
        x = self._inputs['cpu']
        ret = np.where(x > 0, x, (np.exp(x) - 1) * self._alpha)
        self._outputs['cpu'] = ret


class elu_backward(operation):

    name = 'Elu (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward
        self._alpha = self._fwd_op._alpha

    def setup(self, inputs):
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus
        in_shape = inputs.shape
        outs = GraphMultiStorage(shape=in_shape, gpus=gpus)
        self._inputs = inputs
        self._outputs = outs
        self._fwd_in = self._fwd_op._inputs
        self._vars = {'y': outs, id(self._fwd_op._inputs): outs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cueru_backward(self._alpha, self._fwd_in[gpu], self._outputs[gpu])
            rm.cu.cumul(self._outputs[gpu], self._inputs[gpu], self._outputs[gpu], handle)


class elu_backward_cpu(elu_backward):

    def perform(self):
        dy = self._inputs['cpu']
        y = self._fwd_op._outputs['cpu']
        alpha = self._alpha
        ret = np.where(y > 0, dy, (alpha + y) * dy)
        self._outputs['cpu'] = ret


class EluElement(UserGraph):

    def __init__(self, alpha=0.01, previous_elements=None):
        fwd_op = elu_forward(alpha) if rm.is_cuda_active() else elu_forward_cpu(alpha)
        bwd_ops = [elu_backward(fwd_op) if rm.is_cuda_active() else elu_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


class EluGraphElement(GraphFactory):

    def __init__(self, alpha=0.01):
        '''Initializer for Elu graph producing GraphFactory.

            Args:
                alpha (float): Alpha coefficient for Elu.

            For more details, see :class:`.Elu`
        '''
        super().__init__()
        self._alpha = alpha

    def connect(self, other):
        ret = EluElement(alpha=self._alpha, previous_elements=other)
        return ret
