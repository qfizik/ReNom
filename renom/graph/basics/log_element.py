from renom.graph.core import operation, operational_element, UserGraph, GraphMultiStorage, GraphFactory
import renom as rm
import numpy as np


class log_forward(operation):

    name = 'Log (F)'

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
            rm.cuda.culoge(self._inputs[gpu], self._outputs[gpu])


class log_forward_cpu(log_forward):

    def perform(self):
        self._outputs['cpu'] = np.log(self._inputs['cpu'])


class log_backward(operation):

    name = 'Log (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        inputs = inputs[0]['dy']
        fwd_inputs = self._fwd_op._inputs
        shape = fwd_inputs.shape
        gpus = fwd_inputs.gpus

        self.gpus = gpus
        self._inputs = inputs
        self._fwd_inputs = fwd_inputs
        self._outputs = GraphMultiStorage(shape=shape, gpus=gpus)
        self._vars = {'y': self._outputs, 'dy': self._outputs, id(fwd_inputs): self._outputs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            self._outputs[gpu] = self._inputs[gpu] / self._fwd_inputs[gpu]


class log_backward_cpu(log_backward):

    def perform(self):
        self._outputs['cpu'] = self._inputs['cpu'] / self._fwd_inputs['cpu']


class LogElement(UserGraph):

    _name = 'Log Element'

    def __init__(self, previous_element=None):
        fwd_op = log_forward() if rm.is_cuda_active() else log_forward_cpu()
        bwd_ops = [log_backward(fwd_op) if rm.is_cuda_active() else log_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops,
                         previous_elements=previous_element)


def log(self):
    ret = LogElement([self])
    return ret


UserGraph.log = log