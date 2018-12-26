import renom as rm
from renom.graph.core import UserGraph, operation, GraphFactory, graph_variable, GraphMultiStorage
import numpy as np


class _forward_operation(operation):

    def setup(self, inputs):
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus
        self._inputs = inputs

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            dx = self._inputs[gpu]
            y = self._outputs[gpu]
            pass


class _forward_operation_cpu(_forward_operation):

    def perform(self):
        x = self._inputs['cpu']
        # Calculate ret
        self._outputs['cpu'] = ret


class _backward_operation(operation):

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus
        self._inputs = inputs

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            pass


class _backward_operation_cpu(_backward_operation):

    def perform(self):
        dy = self._inputs['cpu']
        # Calculate ret
        self._outputs['cpu'] = ret


class _element(UserGraph):

    has_back = True

    def __init__(self, previous_elements=None):
        fwd_op = _forward_operation() if rm.is_cuda_active() else _forward_operation_cpu()
        bwd_ops = [_backward_operation(fwd_op) if rm.is_cuda_active()
                   else _backward_operation_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


class _graph_element(GraphFactory):

    def __init__(self):
        super().__init__()

    def connect(self, other):
        ret = _element(previous_elements=other)
        return ret
