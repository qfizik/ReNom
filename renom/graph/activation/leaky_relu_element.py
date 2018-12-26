import renom as rm
from renom.graph.core import UserGraph, operation, GraphFactory, graph_variable, GraphMultiStorage
import numpy as np


class leaky_reluforward(operation):

    def __init__(self, slope):
        self._slope = slope

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
            rm.cuda.culeaky_leru_forward(self._slope, self._inputs[gpu], self._outputs[gpu])


class leaky_reluforward_cpu(leaky_reluforward):

    def perform(self):
        x = self._inputs['cpu']
        slope = self._slope
        ret = np.where(x > 0, x, x * slope)
        self._outputs['cpu'] = ret


class leaky_relubackward(operation):

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward
        self._slope = self._fwd_op._slope

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
            rm.cuda.culeaky_leru_backward(self._slope, self._fwd_in[gpu], self._outputs[gpu])
            rm.cu.cumul(self._outputs[gpu], self._inputs[gpu], self._outputs[gpu], handle)


class leaky_relubackward_cpu(leaky_relubackward):

    def perform(self):
        dy = self._inputs['cpu']
        slope = self._slope
        y = self._fwd_op._outputs['cpu']
        ret = np.where(y > 0, dy, dy * slope)
        self._outputs['cpu'] = ret


class LeakyReluElement(UserGraph):

    has_back = True

    def __init__(self, slope=0.01, previous_elements=None):
        fwd_op = leaky_reluforward(slope) if rm.is_cuda_active() else leaky_reluforward_cpu(slope)
        bwd_ops = [leaky_relubackward(fwd_op) if rm.is_cuda_active()
                   else leaky_relubackward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


class LeakyReluGraphElement(GraphFactory):

    def __init__(self, slope=0.01):
        super().__init__()
        self._slope = slope

    def connect(self, other):
        ret = LeakyReluElement(slope=self._slope, previous_elements=other)
        return ret
