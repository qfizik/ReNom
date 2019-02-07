import renom as rm
from renom.graph.core import operation, UserLossGraph, graph_element, GraphMultiStorage, GraphFactory
import numpy as np


class cross_entropy_forward(operation):

    name = 'Cross Entropy (F)'
    roles = ['Loss']

    def setup(self, inputs):
        assert isinstance(inputs[1], dict)

        labels = inputs[1]['y']
        inputs = inputs[0]['y']
        out_shape = (1, )
        gpus = inputs.gpus
        act_out = GraphMultiStorage(shape=inputs.shape, gpus=gpus)
        outs = GraphMultiStorage(shape=out_shape, gpus=gpus)
        self.gpus = gpus
        self._outputs = outs
        self._vars = {'y': outs}
        self._lbls = labels
        self._act_out = act_out
        self._N = inputs.shape[0]
        self._inputs = inputs

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cuadd(self._inputs[gpu], 1e-8, self._act_out[gpu], handle)
            rm.cuda.culoge(self._act_out[gpu], self._act_out[gpu])
            rm.cuda.cumul(self._act_out[gpu], self._lbls[gpu], self._act_out[gpu], handle)
            tmp = rm.cuda.cusum(self._act_out[gpu], handle)
            rm.cuda.cudiv(tmp, self._N, tmp, handle)
            self._outputs[gpu].copy_from(tmp)


class cross_entropy_forward_cpu(cross_entropy_forward):

    def perform(self):
        pred = self._inputs['cpu']
        real = self._lbls['cpu']
        log_pred = np.log(pred + 1e-8)
        ret = -np.sum(real * log_pred).reshape(1)
        self._outputs['cpu'] = ret


class cross_entropy_backward(operation):

    name = 'Cross Entropy (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):

        predictions = inputs[0]['y']
        labels = inputs[1]['y']
        for a, b in zip(predictions.shape, labels.shape):
            assert a == b, '{} / {}'.format(a, b)
        self._N = predictions.shape[0]
        self._graph_input = predictions
        self._label_input = labels

        gpus = predictions.gpus
        self.gpus = gpus
        output = GraphMultiStorage(shape=predictions.shape, gpus=gpus)

        self._outputs = output
        self._vars = {'y': output, 'dy': output, id(self._fwd_op._inputs): output}
        self._N = predictions.shape[0]

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cudiv(self._label_input[gpu],
                          self._graph_input[gpu], self._outputs[gpu], handle)
            rm.cuda.cudiv(self._outputs[gpu], self._N, self._outputs[gpu], handle)


class cross_entropy_backward_cpu(cross_entropy_backward):

    def perform(self):
        pred = self._graph_input['cpu']
        real = self._label_input['cpu']

        ret = -real / pred
        self._outputs['cpu'] = ret


class CrossEntropyElement(UserLossGraph):

    def __init__(self, previous_elements=None):
        fwd_op = cross_entropy_forward() if rm.is_cuda_active() else cross_entropy_forward_cpu()
        bwd_ops = [cross_entropy_backward(fwd_op) if rm.is_cuda_active()
                   else cross_entropy_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


class CrossEntropyGraphElement(GraphFactory):

    def connect(self, predictions, true_values):
        ret = CrossEntropyElement(previous_elements=[predictions, true_values])
        return ret
