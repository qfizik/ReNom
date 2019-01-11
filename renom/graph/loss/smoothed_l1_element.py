import renom as rm
from renom.graph.core import UserLossGraph, operation, GraphMultiStorage, GraphFactory
import numpy as np


class smoothed_l1_forward(operation):

    name = 'Mean Squared (F)'
    roles = ['Loss']

    def __init__(self, delta=1.0):
        self._delta = delta

    def setup(self, inputs):
        predictions = inputs[0]['y']
        real_values = inputs[1]['y']
        self.gpus = predictions.gpus
        self._graph_input = predictions
        self._label_input = real_values

        out_shape = (1, )
        assert predictions.shape == real_values.shape
        output = GraphMultiStorage(shape=out_shape, gpus=predictions.gpus)

        self._vars = {'y': output}
        self._outputs = output
        self._N = predictions.shape[0]

    def perform(self):
        self._d = GraphMultiStorage(shape=self._graph_input.shape, gpus=self.gpus)
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            x = self._graph_input[gpu].new_array()
            y = self._label_input[gpu].new_array()
            N = len(x)
            d = x - y
            delta = self._delta
            abs_d = abs(d)
            flag = abs_d < delta
            ret = np.sum(flag * 0.5 * (d * d) +
                         (1 - flag) * (abs_d - 0.5 * delta) * delta)
            ret = ret.reshape(1,) / N
            self._d[gpu] = d
            self._outputs[gpu].to_gpu(ret)


class smoothed_l1_forward_cpu(smoothed_l1_forward):

    def perform(self):
        x = self._graph_input['cpu']
        y = self._label_input['cpu']
        N = len(x)
        d = x - y
        delta = self._delta
        abs_d = abs(d)
        flag = abs_d < delta
        ret = np.sum(flag * 0.5 * (d * d) +
                     (1 - flag) * (abs_d - 0.5 * delta) * delta)
        ret = ret.reshape(1,) / N
        self._d = d
        self._outputs['cpu'] = ret


class smoothed_l1_backward(operation):

    name = 'Mean Squared (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward
        self._delta = self._fwd_op._delta

    def setup(self, inputs):

        predictions = inputs[0]['y']
        real_values = inputs[1]['y']
        self._graph_input = predictions
        self._label_input = real_values
        gpus = predictions.gpus
        self.gpus = gpus
        output = GraphMultiStorage(shape=predictions.shape, gpus=gpus)
        self._outputs = output
        self._vars = {'y': output, 'dy': output, id(self._fwd_op._graph_input): output}
        self._N = predictions.shape[0]

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            d = self._fwd_op._d[gpu]
            N = len(d)
            delta = self._delta
            mask = abs(d) <= delta
            sign = (d > 0) * 2 - 1
            dx = mask * d + (1 - mask) * sign * delta
            ret = dx / N
            self._outputs[gpu].to_gpu(ret)


class smoothed_l1_backward_cpu(smoothed_l1_backward):

    def perform(self):
        d = self._fwd_op._d
        N = len(d)
        delta = self._delta
        mask = abs(d) <= delta
        sign = (d > 0) * 2 - 1
        dx = mask * d + (1 - mask) * sign * delta
        ret = dx / N
        self._outputs['cpu'] = ret


class SmoothedL1Element(UserLossGraph):

    def __init__(self, delta=1.0, previous_elements=None):
        self._delta = delta
        fwd_op = smoothed_l1_forward(
            delta) if rm.is_cuda_active() else smoothed_l1_forward_cpu(delta)
        bwd_ops = [smoothed_l1_backward(fwd_op) if rm.is_cuda_active()
                   else smoothed_l1_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


class SmoothedL1GraphElement(GraphFactory):

    def __init__(self, delta=1.0):
        super().__init__()
        self._delta = delta

    def connect(self, predictions, true_values):
        ret = SmoothedL1Element(self._delta, previous_elements=[predictions, true_values])
        return ret
