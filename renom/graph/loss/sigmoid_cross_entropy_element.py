import renom as rm
from renom.graph.core import operation, UserLossGraph, graph_element, GraphMultiStorage, GraphFactory
import numpy as np


class sigmoid_forward(operation):

    name = 'Sigmoid Cross Entropy(F)'
    roles = ['loss']

    def setup(self, inputs):
        assert isinstance(inputs[1], dict)

        labels = inputs[1]['y']
        inputs = inputs[0]['y']
        assert labels.shape == inputs.shape
        out_shape = (1, )
        gpus = inputs.gpus
        outs = GraphMultiStorage(shape=out_shape, gpus=gpus)
        self.gpus = gpus
        self._outputs = outs
        self._vars = {'y': outs}
        self._lbls = labels
        self._N = inputs.shape[0]
        self._inputs = inputs

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            x = self._inputs[gpu]
            N = x.shape[0]
            y = self._lbls[gpu]
            tmp1 = x.empty_like_me()
            tmp2 = x.empty_like_me()
            tmp3 = x.empty_like_me()
            rm.cuda.cusigmoid(x, tmp1)
            rm.cuda.cucross_entropy(tmp1, y, tmp2, handle)
            rm.cuda.cucross_entropy(-tmp1 + 1, -y + 1, tmp3, handle)
            tmp = rm.cuda.cusum(-(tmp2 + tmp3), handle)
            self._outputs[gpu].copy_from(tmp)
            rm.cuda.cudiv(self._outputs[gpu], N, self._outputs[gpu], handle)


class sigmoid_forward_cpu(sigmoid_forward):

    def perform(self):
        x = self._inputs['cpu']
        y = self._lbls['cpu']
        N = len(x)
        z = 1 / (1 + np.exp(-x))
        self._z = z
        ret = -np.sum(y * np.log(z + 1e-8) + (1 - y) * np.log(1 - z + 1e-8)).reshape(1) / N
        self._outputs['cpu'] = ret.reshape(1,)


class sigmoid_backward(operation):

    name = 'Sigmoid Cross Entropy(B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):

        if len(inputs) > 3:
            self._dy = inputs[3]['y']
        else:
            self._dy = None
        predictions = inputs[0]['y']
        labels = inputs[1]['y']
        self._N = predictions.shape[0]
        self._graph_input = predictions
        self._label_input = labels

        gpus = predictions.gpus
        self.gpus = gpus
        output = GraphMultiStorage(shape=predictions.shape, gpus=gpus)
        act_out1 = GraphMultiStorage(shape=predictions.shape, gpus=gpus)

        self._act_out1 = act_out1
        self._outputs = output
        self._vars = {'y': output, 'dy': output, id(self._fwd_op._inputs): output}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            if self._dy is not None:
                dy = self._dy[gpu]
            else:
                dy = 1
            rm.cuda.cusigmoid(self._graph_input[gpu], self._act_out1[gpu])
            rm.cuda.cusub(self._act_out1[gpu], self._label_input[gpu], self._outputs[gpu], handle)
            rm.cuda.cudiv(self._outputs[gpu], self._N, self._outputs[gpu], handle)
            rm.cuda.cumul(self._outputs[gpu], dy, self._outputs[gpu], handle)



class sigmoid_backward_cpu(sigmoid_backward):

    def perform(self):
        z = self._fwd_op._z
        y = self._label_input['cpu']
        N = len(z)
        if self._dy is not None:
            dy = self._dy['cpu']
        else:
            dy = 1
        ret = (z - y) * dy / N
        self._outputs['cpu'] = ret


class SigmoidCrossEntropyElement(UserLossGraph):

    def __init__(self, previous_elements=None):
        fwd_op = sigmoid_forward() if rm.is_cuda_active() else sigmoid_forward_cpu()
        bwd_ops = [sigmoid_backward(fwd_op) if rm.is_cuda_active()
                   else sigmoid_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


class SigmoidCrossEntropy(GraphFactory):

    def connect(self, predictions, true_values):
        ret = SigmoidCrossEntropyElement(previous_elements=[predictions, true_values])
        return ret
