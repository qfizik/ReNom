import renom as rm
from renom.graph.core import UserGraph, operation, GraphFactory, graph_variable, GraphMultiStorage
import numpy as np


class relu_forward(operation):

    name = 'Relu (F)'

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
            rm.cuda.curelu_foward(self._inputs[gpu], self._outputs[gpu])


class relu_forward_cpu(relu_forward):

    def perform(self):
        x = self._inputs['cpu']
        ret = np.maximum(x, 0)
        self._outputs['cpu'] = ret


class relu_backward(operation):

    name = 'Relu (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        inputs = inputs[0]['dy']
        gpus = inputs.gpus
        self.gpus = gpus
        in_shape = inputs.shape
        outs = GraphMultiStorage(shape=in_shape, gpus=gpus)
        self._inputs = inputs
        self._outputs = outs
        self._fwd_in = self._fwd_op._inputs
        self._vars = {'y': outs, 'dy': outs, id(
            self._fwd_op._inputs): outs, id(self._fwd_in): self._fwd_in}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.curelu_backard(self._fwd_in[gpu], self._outputs[gpu])
            rm.cu.cumul(self._outputs[gpu], self._inputs[gpu], self._outputs[gpu], handle)


class relu_backward_cpu(relu_backward):

    def perform(self):
        dy = self._inputs['cpu']
        y = self._fwd_op._outputs['cpu']
        ret = np.where(y == 0, 0, dy)
        self._outputs['cpu'] = ret


class ReluElement(UserGraph):

    def __init__(self, previous_elements=None):
        fwd_op = relu_forward() if rm.is_cuda_active() else relu_forward_cpu()
        bwd_ops = [relu_backward(fwd_op) if rm.is_cuda_active() else relu_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


class Relu(GraphFactory):

    def connect(self, other):
        ret = ReluElement(previous_elements=other)
        return ret

def relu(x):
    return ReluElement(previous_elements=[x])
