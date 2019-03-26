import renom as rm
from renom.graph.core import UserLossGraph, operation, GraphMultiStorage, GraphFactory
import numpy as np


class mean_squared_forward(operation):

    name = 'Mean Squared (F)'
    roles = ['loss']

    def __init__(self, reduction):
        self.reduction = reduction

    def setup(self, inputs):
        predictions = inputs[0]['y']
        real_values = inputs[1]['y']
        self.gpus = predictions.gpus
        self._graph_input = predictions
        self._label_input = real_values

        self._N = predictions.shape[0]
        assert predictions.shape == real_values.shape

        out_shape = predictions.shape if self.reduction is None else (1, )
        tmp = GraphMultiStorage(shape=predictions.shape, gpus=self.gpus)
        output = GraphMultiStorage(shape=out_shape, gpus=predictions.gpus)

        self._vars = {'y': output}
        self._outputs = output
        self._N = predictions.shape[0]
        self._tmp = tmp

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cusub(self._graph_input[gpu], self._label_input[gpu], self._tmp[gpu], handle)
            rm.cuda.cumul(self._tmp[gpu], self._tmp[gpu], self._tmp[gpu], handle)
            if self.reduction is None:
                self._outputs[gpu].copy_from(self._tmp[gpu])
            else:
                tmp = rm.cuda.cusum(self._tmp[gpu], handle)
                if self.reduction == 'mean':
                    rm.cuda.cudiv(tmp, int(self._N), tmp, handle)
                elif self.reduction == 'sum':
                    pass
                else:
                    pass
                self._outputs[gpu].copy_from(tmp)


class mean_squared_forward_cpu(mean_squared_forward):

    def perform(self):
        pred = self._graph_input['cpu']
        real = self._label_input['cpu']
        N = len(pred)
        ret = np.square(pred - real) / 2.
        if self.reduction is None:
            pass
        else:
            ret = np.sum(ret).reshape(1,)
            if self.reduction == 'mean':
                ret /= N
            elif self.reduction == 'sum':
                pass
            else:
                pass
        self._outputs['cpu'] = ret


class mean_squared_backward(operation):

    name = 'Mean Squared (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        self.reduction = self._fwd_op.reduction
        predictions = inputs[0]['y']
        real_values = inputs[1]['y']
        if len(inputs) > 3:
            self._dy = inputs[3]['y']
        else:
            self._dy = None
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
            if self._dy is not None:
                dy = self._dy[gpu]
            else:
                dy = 1
            rm.cuda.cusub(self._graph_input[gpu],
                          self._label_input[gpu], self._outputs[gpu], handle)
            rm.cuda.cumul(self._outputs[gpu], 2, self._outputs[gpu], handle)
            if self.reduction is None:
                pass
            else:
                if self.reduction == 'mean':
                    rm.cuda.cudiv(self._outputs[gpu], int(self._N), self._outputs[gpu], handle)
                elif self.reduction == 'sum':
                    pass
                else:
                    pass
            rm.cuda.cumul(self._outputs[gpu], dy, self._outputs[gpu], handle)


class mean_squared_backward_cpu(mean_squared_backward):

    def perform(self):
        #dy = self._inputs['cpu']
        pred = self._graph_input['cpu']
        real = self._label_input['cpu']
        N = len(pred)
        if self._dy is not None:
            dy = self._dy['cpu']
        else:
            dy = 1
        tmp = pred - real
        if self.reduction is None:
            pass
        else:
            if self.reduction == 'mean':
                tmp /= N
            elif self.reduction == 'sum':
                pass
            else:
                pass
        self._outputs['cpu'] = tmp * dy


class MeanSquaredElement(UserLossGraph):

    def __init__(self, reduction='mean', previous_elements=None):

        fwd_op = mean_squared_forward(reduction) if rm.is_cuda_active(
        ) else mean_squared_forward_cpu(reduction)
        bwd_ops = [mean_squared_backward(fwd_op) if rm.is_cuda_active()
                   else mean_squared_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


class MeanSquared(GraphFactory):
    """A factory class of mean squared loss function element.

    .. math::
        target, x \in R^{N \\times D} \\\\
        y = \\frac{1}{N} \sum_{n, d}{(x_{nd} - target_{nd})^2}

    +-----------+-------------------------------------------------------+
    | reduction |  description                                          |
    +===========+=======================================================+
    |  'mean'   | Calculates mean along axis=0 then sum up all element. |
    +-----------+-------------------------------------------------------+
    |  'sum'    | Calculates sum of all element.                        |
    +-----------+-------------------------------------------------------+
    |   None    | Reduction is not performed.                           |
    +-----------+-------------------------------------------------------+
    """

    def prepare(self, reduction='mean'):
        self.reduction = reduction

    def connect(self, predictions, true_values):
        ret = MeanSquaredElement(self.reduction, previous_elements=[predictions, true_values])
        return ret
