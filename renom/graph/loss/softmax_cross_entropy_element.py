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

class softmax_cross_entropy_forward(operation):

    name = 'Softmax Cross Entropy(F)'
    roles = ['loss']

    def __init__(self, reduction):
        self.reduction = reduction

    def setup(self, inputs):
        assert isinstance(inputs[1], dict)
        labels = inputs[1]['y']
        inputs = inputs[0]['y']
        out_shape = predictions.shape if self.reduction is None else (1, )
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
            rm.cuda.cuSoftmaxForward(handle, self._inputs[gpu], self._act_out[gpu], mode=1)
            rm.cuda.cucross_entropy(self._act_out[gpu], self._lbls[gpu], self._act_out[gpu], handle)

            rm.cuda.cunegate(self._act_out[gpu], self._act_out[gpu])
            if self.reduction is None:
                self._outputs[gpu].copy_from(self._act_out[gpu])
            else:
                tmp = rm.cuda.cusum(self._act_out[gpu], handle)
                if self.reduction == 'mean':
                    rm.cuda.cudiv(tmp, self._N, tmp, handle)
                elif self.reduction == 'sum':
                    pass
                else:
                    pass
                self._outputs[gpu].copy_from(tmp)

    def get_loss(self):
        loss = 0
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            arr = np.empty(self._outputs.shape, dtype=rm.precision)
            pin = handle.getPinnedMemory(arr)
            self._outputs[gpu].to_cpu(pin)
            pin.unpin(arr)
            loss += arr
        return loss


class softmax_cross_entropy_forward_cpu(softmax_cross_entropy_forward):

    def perform(self):
        x = self._inputs['cpu']
        y = self._lbls['cpu']
        assert x.shape == y.shape
        N = len(x)
        maxes = np.max(x, axis=1, keepdims=True)
        u = np.exp(x - maxes)
        summed = np.sum(u, axis=1, keepdims=True)
        z = u / (summed + 1e-8)
        self._z = z
        ret = -(y * np.log(z + 1e-8))
        if self.reduction is None:
            pass
        else:
            ret = np.sum(ret).reshape(1,)
            if self.reduction == 'mean':
                if N > 0:
                    ret /= N
            elif self.reduction == 'sum':
                pass
            else:
                pass
        self._outputs['cpu'] = ret


class softmax_cross_entropy_backward(operation):

    name = 'Softmax Cross Entropy(B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
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
        self._vars = {'y': output, id(self._fwd_op._inputs): output}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            if self._dy is not None:
                dy = self._dy[gpu]
            else:
                dy = 1
            rm.cuda.cuSoftmaxForward(handle, self._graph_input[gpu], self._outputs[gpu], mode=1)
            rm.cuda.cusub(self._outputs[gpu], self._label_input[gpu], self._outputs[gpu], handle)

            if self.reduction is not None:
                if self.reduction == 'mean':
                    rm.cuda.cudiv(self._outputs[gpu], self._N, self._outputs[gpu], handle)

            rm.cuda.cumul(self._outputs[gpu], dy, self._outputs[gpu], handle)


class softmax_cross_entropy_backward_cpu(softmax_cross_entropy_backward):

    def perform(self):
        if self._dy is not None:
            dy = self._dy['cpu']
        else:
            dy = 1
        y = self._label_input['cpu']
        z = self._fwd_op._z
        ret = z - y
        if self.reduction is not None:
            if self.reduction == 'mean':
                N = len(z)
                ret /= N
        self._outputs['cpu'] = ret * dy


class SoftmaxCrossEntropyElement(UserLossGraph):

    def __init__(self, reduction='mean', previous_elements=None):
        fwd_op = softmax_cross_entropy_forward(reduction) if rm.is_cuda_active(
        ) else softmax_cross_entropy_forward_cpu(reduction)
        bwd_ops = [softmax_cross_entropy_backward(
            fwd_op) if rm.is_cuda_active() else softmax_cross_entropy_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


@populate_graph
class SoftmaxCrossEntropy(GraphFactory):

    '''
    Softmax cross entropy.

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

    def connect(self, predictions, true_values):
        ret = SoftmaxCrossEntropyElement(reduction=self.reduction,
                                         previous_elements=[predictions, true_values])
        return ret
