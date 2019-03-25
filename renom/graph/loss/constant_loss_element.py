
import numpy as np
import renom as rm
from renom.graph.core import operation, operational_element, UserLossGraph, GraphMultiStorage, GraphFactory
from renom.graph.basics.sum_element import sum_forward, sum_forward_cpu
import renom.utility.initializer as init


class constant_loss_forward(operation):

    name = 'Constant loss'

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def setup(self, inputs):
        inputs = inputs[0]['y']
        self.gpus = inputs.gpus
        output_shape = inputs.shape if self.reduction is None else (1, )
        self._inputs = inputs
        self._outputs = GraphMultiStorage(shape=output_shape, gpus=self.gpus)
        self._vars = {'y': self._outputs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            if self.reduction is None:
                self._outputs[gpu] = self._inputs[gpu]

            elif self.reduction == 'sum':
                r = rm.cuda.cusum(self._inputs[gpu], handle, axis=None, keepdims=False)
                self._outputs[gpu].copy_from(r)

            elif self.reduction == 'mean':
                N = len(self._inputs[gpu])
                r = rm.cuda.cusum(self._inputs[gpu], handle, axis=None, keepdims=False)
                rm.cuda.cudiv(r, N, self._outputs[gpu], handle)


class constant_loss_forward_cpu(constant_loss_forward):

    def perform(self):
        if self.reduction is None:
            self._outputs['cpu'] = self._inputs['cpu']

        elif self.reduction == 'sum':
            self._outputs['cpu'] = np.array(np.sum(self._inputs['cpu']))

        elif self.reduction == 'mean':
            N = getattr(self._inputs['cpu'], '__len__', lambda: 1)()
            self._outputs['cpu'] = np.array(np.sum(self._inputs['cpu']) / N)


class constant_loss_backward(operation):

    name = 'Constant (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        self.reduction = self._fwd_op.reduction

        if len(inputs) > 2:
            self._dy = inputs[2]['y']
        else:
            self._dy = None

        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus

        outputs = GraphMultiStorage(shape=inputs.shape, gpus=gpus, initializer=init.Constant(1))
        self._fwd_inputs = self._fwd_op._inputs
        self._outputs = outputs
        self._vars = {'y': outputs}

    def perform(self):
        shape = self._outputs.shape
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            if self._dy is None:
                if self.reduction == 'mean':
                    N = int(shape[0])
                    rm.cuda.cufill(1. / N, self._outputs[gpu], handle)
                elif self.reduction == 'sum':
                    rm.cuda.cufill(1., self._outputs[gpu], handle)
            else:
                if self.reduction == 'mean':
                    N = int(shape[0])
                    rm.cuda.cufill(1. / N, self._outputs[gpu], handle)
                elif self.reduction == 'sum':
                    rm.cuda.cufill(1., self._outputs[gpu], handle)
                rm.cuda.cumul(self._outputs[gpu], self._dy[gpu], self._outputs[gpu], handle)


class constant_loss_backward_cpu(constant_loss_backward):

    def perform(self):
        shape = self._outputs.shape
        if self._dy is not None:
            dy = self._dy['cpu']
            if self.reduction == 'mean':
                self._outputs['cpu'][:] = np.ones(shape, dtype=rm.precision)
            elif self.reduction == 'sum':
                self._outputs['cpu'][:] = np.ones(shape, dtype=rm.precision)
            self._outputs['cpu'] *= dy
        else:
            if self.reduction == 'mean':
                self._outputs['cpu'][:] = np.ones(shape, dtype=rm.precision)
            elif self.reduction == 'sum':
                self._outputs['cpu'][:] = np.ones(shape, dtype=rm.precision)


class ConstantLossElement(UserLossGraph):

    is_connector_element = True

    def __init__(self, reduction='mean', previous_element=None):
        fwd_op = constant_loss_forward(reduction) if rm.is_cuda_active(
        ) else constant_loss_forward_cpu(reduction)

        fwd_op.roles = ['loss']
        bwd_ops = [constant_loss_backward(fwd_op) if rm.is_cuda_active()
                   else constant_loss_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_element)
        self._bwd_graphs[0].add_input(previous_element.get_forward_output())
        self._bwd_graphs[0].add_input(self._fwd)


class ConstantLoss(GraphFactory):
    '''A factory class of constant loss function element.

    Agrs:
        reduction (str, None): Reduction method. Available arguments are described below.

    +-----------+-------------------------------------------------------+
    | reduction |  description                                          |
    +===========+=======================================================+
    |  'mean'   | Calculates mean along axis=0 then sum up all element. |
    +-----------+-------------------------------------------------------+
    |  'sum'    | Calculates sum of all element.                        |
    +-----------+-------------------------------------------------------+
    |   None    | Reduction is not performed.                           |
    +-----------+-------------------------------------------------------+
    '''

    def prepare(self, reduction='mean'):
        self.reduction = reduction

    def connect(self, other):
        return ConstantLossElement(self.reduction, previous_element=other)
