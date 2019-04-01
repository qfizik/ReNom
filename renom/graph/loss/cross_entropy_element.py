#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

import numpy as np

import renom as rm
from renom.graph.core import operation, UserLossGraph, graph_element, GraphMultiStorage, GraphFactory
from renom.graph import populate_graph


class cross_entropy_forward(operation):
    '''Cross Entropy forward operation class.
    '''

    name = 'Cross Entropy (F)'
    roles = ['loss']

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def setup(self, inputs):
        '''Prepares workspaces for this operation.

        Args:
            inputs (list of GraphMultiStorage): Input data to this operation.

        constant_loss_forward class requires inputs to contain following keys.

        +-------+-----+------------------------------------+
        | Index | Key |              Role                  |
        +=======+=====+====================================+
        |   0   |  y  | Output of forward propagation.     |
        +-------+-----+------------------------------------+
        |   1   |  y  | Target of associated input.        |
        +-------+-----+------------------------------------+
        '''

        assert isinstance(inputs[1], dict)

        labels = inputs[1]['y']
        inputs = inputs[0]['y']

        gpus = inputs.gpus
        out_shape = inputs.shape if self.reduction is None else (1, )
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
            if self.reduction is not None:
                N = len(self._act_out[gpu])
                tmp = rm.cuda.cusum(self._act_out[gpu], handle)
                if self.reduction == 'mean':
                    rm.cuda.cudiv(tmp, N, tmp, handle)
                elif self.reduction == 'sum':
                    pass
                else:
                    pass
            self._outputs[gpu].copy_from(tmp)


class cross_entropy_forward_cpu(cross_entropy_forward):

    def perform(self):
        pred = self._inputs['cpu']
        real = self._lbls['cpu']
        log_pred = -np.log(pred + 1e-8)
        ret = real * log_pred
        if self.reduction is None:
            pass
        else:
            ret = np.sum(ret).reshape(1)
            if self.reduction == 'mean':
                N = len(pred)
                ret /= N
            elif self.reduction == 'sum':
                pass
            else:
                pass
        self._outputs['cpu'] = ret


class cross_entropy_backward(operation):
    '''Cross Entropy backward operation class.
    '''

    name = 'Cross Entropy (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        '''Prepares workspaces for this operation.

        Args:
            inputs (list of GraphMultiStorage): Input data to this operation.

        cross_entropy_backward class requires inputs to contain following keys.

        +-------+-----+------------------------------------+
        | Index | Key |              Role                  |
        +=======+=====+====================================+
        |   0   |  y  | Output of forward propagation.     |
        +-------+-----+------------------------------------+
        |   1   |  y  | Target associated to the input.    |
        +-------+-----+------------------------------------+
        |   2   |  y  | Output of forward operation.       |
        +-------+-----+------------------------------------+
        |   3   |  y  | Output of previous operation.      |
        +-------+-----+------------------------------------+
        '''

        self.reduction = self._fwd_op.reduction
        if len(inputs) > 3:
            self._dy = inputs[3]['y']
        else:
            self._dy = None

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
            if self._dy is not None:
                dy = self._dy[gpu]
            else:
                dy = 1
            rm.cuda.cudiv(self._label_input[gpu],
                          self._graph_input[gpu], self._outputs[gpu], handle)
            rm.cuda.cumul(self._outputs[gpu], dy, self._outputs[gpu], handle)

            if self.reduction == 'mean':
                N = len(self._graph_input[gpu])
                rm.cuda.cudiv(self._outputs[gpu], N, self._outputs[gpu], handle)


class cross_entropy_backward_cpu(cross_entropy_backward):

    def perform(self):
        pred = self._graph_input['cpu']
        real = self._label_input['cpu']
        if self._dy is not None:
            dy = self._dy['cpu']
        else:
            dy = 1
        ret = -real * dy / pred

        if self.reduction == 'mean':
            N = len(pred)
            ret /= N

        self._outputs['cpu'] = ret


class CrossEntropyElement(UserLossGraph):

    def __init__(self, reduction='mean', previous_elements=None):
        fwd_op = cross_entropy_forward(reduction) if rm.is_cuda_active(
        ) else cross_entropy_forward_cpu(reduction)
        bwd_ops = [cross_entropy_backward(fwd_op) if rm.is_cuda_active()
                   else cross_entropy_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


@populate_graph
class CrossEntropy(GraphFactory):
    '''A factory class of cross entropy loss function element.

    .. math::

        target, x \in R^{N \\times D} \\\\
        y = -\\frac{1}{N}\sum_{n}{\sum_{d}{target_{nd} * log(x_{nd})}}

    Args:
        reduction (str): Reduction method. This accepts following keywords.

    +-----------+-------------------------------------------------------+
    | reduction |  description                                          |
    +===========+=======================================================+
    |  'mean'   | Calculates mean along axis=0 then sum up all element. |
    +-----------+-------------------------------------------------------+
    |  'sum'    | Calculates sum of all element.                        |
    +-----------+-------------------------------------------------------+
    |   None    | Reduction is not performed.                           |
    +-----------+-------------------------------------------------------+

    Example:
        >>> import numpy as np
        >>> import renom.graph as rmg
        >>>
        >>> v1 = rmg.StaticVariable(np.random.rand(2, 2))
        >>> v2 = rmg.StaticVariable(np.random.rand(2, 2))
        >>>
        >>> rmg.cross_entropy(v1, v2)
        Cross Entropy (F):
        [1.1589496]
    '''

    def prepare(self, reduction='mean'):
        self.reduction = reduction

    def connect(self, predictions, true_values):
        ret = CrossEntropyElement(self.reduction, previous_elements=[predictions, true_values])
        return ret

@populate_graph
def cross_entropy(x, y, reduction='mean'):
    '''A function style factory of cross entropy operation element.

    Args:
        x (UserGraph, ndarray): Left hand input.
        y (UserGraph, ndarray): Right hand input.
        reduction (str, None): Reduction method.


    For more information, please refer :py:class:`~renom.graph.loss.cross_entropy_element.CrossEntropy`.
    '''

    return CrossEntropy(reduction=reduction)(x, y)
