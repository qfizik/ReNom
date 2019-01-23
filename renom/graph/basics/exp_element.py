from renom.graph.core import operation, operational_element, UserGraph, GraphMultiStorage, GraphFactory
import renom as rm
import numpy as np


class exp_forward(operation):

    name = 'Exp (F)'

    def setup(self, inputs):
        inputs = inputs[0]['y']
        output_shape = inputs.shape
        gpus = inputs.gpus

        self._inputs = inputs
        self.gpus = gpus
        self._outputs = GraphMultiStorage(shape=output_shape, gpus=gpus)
        self._vars = {'y': self._outputs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cuexp(self._inputs[gpu], self._outputs[gpu])


class exp_forward_cpu(exp_forward):

    def perform(self):
        self._outputs['cpu'] = np.exp(self._inputs['cpu'])


class exp_backward(operation):

    name = 'Exp (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        inputs = inputs[0]['dy']
        fwd_inputs = self._fwd_op._inputs
        shape = fwd_inputs.shape
        gpus = fwd_inputs.gpus

        self.gpus = gpus
        self._inputs = inputs
        self._fwd_outputs = self._fwd_op._outputs
        self._outputs = GraphMultiStorage(shape=shape, gpus=gpus)
        self._vars = {'y': self._outputs, 'dy': self._outputs, id(fwd_inputs): self._outputs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            self._outputs[gpu] = self._fwd_outputs[gpu] * self._inputs[gpu]


class exp_backward_cpu(exp_backward):

    def perform(self):
        self._outputs['cpu'] = self._fwd_outputs['cpu'] * self._inputs['cpu']


class ExpElement(UserGraph):

    _name = 'Exp Element'

    def __init__(self, previous_element=None):
        fwd_op = exp_forward() if rm.is_cuda_active() else exp_forward_cpu()
        bwd_ops = [exp_backward(fwd_op) if rm.is_cuda_active() else exp_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops,
                         previous_elements=previous_element)


def exp(self):
    ret = ExpElement([self])
    return ret


UserGraph.exp = exp
