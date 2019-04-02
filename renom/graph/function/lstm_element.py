#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

import numpy as np

import renom as rm
from renom.graph.core import UserGraph, operation, GraphFactory, graph_variable, GraphMultiStorage
from renom.graph.train import initializer as init
from renom.graph import populate_graph


class lstm_forward(operation):

    name = 'LSTM (F)'
    consumes = ['w', 'wr']

    def __init__(self, output_size, initializer=None):
        self._output_size = output_size
        self._init = init.GlorotNormal() if initializer is None else initializer

    def setup(self, inputs):
        if len(inputs) > 3:
            prev_lstm = inputs[3]
        else:
            prev_lstm = None

        weights_r = inputs[2]['y']
        weights = inputs[1]['y']
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus
        self._inputs = inputs
        input_shape = inputs.shape

        weight_shape = (input_shape[1], self._output_size * 4)
        weight_r_shape = (self._output_size, self._output_size * 4)
        out_shape = (input_shape[0], self._output_size)
        weights.__init__(shape=weight_shape, gpus=self.gpus, initializer=self._init)
        weights_r.__init__(shape=weight_r_shape, gpus=self.gpus, initializer=self._init)
        outs = GraphMultiStorage(shape=out_shape, gpus=self.gpus)
        self._vars = {'y': outs, 'w': weights, 'wr': weights_r, 'pfgate': None}
        self._weights = weights
        self._weights_r = weights_r
        self._outputs = outs
        self._prev = prev_lstm

    def perform(self):
        prev = self._prev
        n, m = self._inputs.shape[0], self._weights.shape[1] // 4
        new_s = GraphMultiStorage(shape=(n, m), gpus=self.gpus)
        new_z = GraphMultiStorage(shape=(n, m), gpus=self.gpus)
        new_u = GraphMultiStorage(shape=(n, m * 4), gpus=self.gpus)
        if prev is not None:
            ps = prev['s']
            pz = prev['z']
        else:
            ps = GraphMultiStorage(shape=(n, m), gpus=self.gpus, initializer=init.Constant(0))
            pz = GraphMultiStorage(shape=(n, m), gpus=self.gpus, initializer=init.Constant(0))
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            x = self._inputs[gpu]
            w = self._weights[gpu]
            wr = self._weights_r[gpu]

            tmp1 = rm.GPUValue(shape=(x.shape[0], w.shape[1]))
            tmp2 = rm.GPUValue(shape=(x.shape[0], w.shape[1]))

            rm.cuda.cublas_gemm(x, 0, w, 0, tmp1, handle)
            rm.cuda.cublas_gemm(pz[gpu], 0, wr, 0, tmp2, handle)

            #u = tmp1 + tmp2
            rm.cuda.cuadd(tmp1, tmp2, new_u[gpu], handle)
            rm.cuda.culstm_forward_activate(new_u[gpu])
            rm.cuda.culstm_forward(new_u[gpu], new_s[gpu], ps[gpu], new_z[gpu])

            ret = new_z[gpu]

            # Calculate ret
            self._outputs[gpu] = ret

        if prev is not None:
            prev['pfgate'] = new_u
        self._vars['s'] = new_s
        self._vars['ps'] = ps
        self._vars['u'] = new_u
        self._vars['x'] = self._inputs
        self._vars['z'] = self._outputs
        self._vars['pz'] = pz


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class lstm_forward_cpu(lstm_forward):

    def perform(self):
        prev = self._prev
        x = self._inputs['cpu']
        w = self._weights['cpu']
        wr = self._weights_r['cpu']
        if prev is not None:
            s = prev['s']
            z = prev['z']
        else:
            s = np.zeros((x.shape[0], w.shape[1] // 4), dtype=rm.precision)
            z = np.zeros((x.shape[0], w.shape[1] // 4), dtype=rm.precision)

        u = np.dot(x, w) + np.dot(z, wr)
        m = u.shape[1] // 4
        u, gated = np.split(u, [m, ], axis=1)
        u = np.tanh(u)

        gated = sigmoid(gated)

        state = gated[:, m:m * 2] * u + gated[:, :m] * s
        ret = np.tanh(state) * gated[:, m * 2:]

        # Calculate ret
        self._outputs['cpu'] = ret
        if prev is not None:
            prev['pfgate'] = gated[:, :m]
        self._vars['s'] = state
        self._vars['ps'] = s
        self._vars['u'] = u
        self._vars['gate'] = gated
        self._vars['x'] = x
        self._vars['z'] = ret


class lstm_backward(operation):

    name = 'LSTM (B)'
    produces = ['w', 'wr']

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        self._inputs = inputs
        gpus = inputs[0]['y'].gpus
        self.gpus = gpus
        self._inputs = inputs
        outs = GraphMultiStorage(shape=self._fwd_op._inputs.shape, gpus=self.gpus)
        w_out = GraphMultiStorage(shape=self._fwd_op._weights.shape, gpus=self.gpus)
        w_r_out = GraphMultiStorage(shape=self._fwd_op._weights_r.shape, gpus=self.gpus)
        self._outputs = outs
        self._w_out = w_out
        self._w_r_out = w_r_out
        self._vars = {'y': outs, id(self._fwd_op._inputs): outs,
                      'w': w_out, id(self._fwd_op._weights): w_out,
                      'wr': w_r_out, id(self._fwd_op._weights_r): w_r_out,
                      'pfgate': None
                      }

    def perform(self):
        fwd = self._fwd_op._vars
        drt = None
        dou = None

        w = fwd['w']
        wr = fwd['wr']

        u = fwd['u']
        s = fwd['s']
        ps = fwd['ps']

        pfg = fwd['pfgate']
        if pfg is None:
            pfg = GraphMultiStorage(shape=u.shape, gpus=self.gpus, initializer=init.Constant(0))

        for grad in self._inputs:
            if 'pfgate' in grad:
                drt = grad['drt']
                dou = grad['dou']
        if drt is None:
            drt = GraphMultiStorage(shape=u.shape, gpus=self.gpus, initializer=init.Constant(0))
            dou = GraphMultiStorage(shape=fwd['y'].shape,
                                    gpus=self.gpus, initializer=init.Constant(0))

        dr = GraphMultiStorage(shape=u.shape, gpus=self.gpus)
        dou_n = GraphMultiStorage(shape=fwd['y'].shape, gpus=self.gpus)
        new_z = GraphMultiStorage(shape=fwd['y'].shape, gpus=self.gpus)
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            dy = fwd['y'][gpu].zeros_like_me()
            for grad in self._inputs:
                if 'pfgate' in grad:
                    dy += grad['z'][gpu]
                else:
                    dy += grad['y'][gpu]

            rm.cuda.cutanh(s[gpu], s[gpu])
            rm.cuda.culstm_backward(u[gpu], dr[gpu], s[gpu], ps[gpu],
                                    dy, pfg[gpu], dou[gpu], dou_n[gpu])

            # dx
            rm.cuda.cublas_gemm(dr[gpu], 0, w[gpu], 1, self._outputs[gpu], handle)

            # dw
            rm.cuda.cublas_gemm(fwd['x'][gpu], 1, dr[gpu], 0, self._w_out[gpu], handle)

            # dwr
            rm.cuda.cublas_gemm(fwd['y'][gpu], 1, drt[gpu], 0, self._w_r_out[gpu], handle)

            # dz
            rm.cuda.cublas_gemm(dr[gpu], 0, wr[gpu], 1, new_z[gpu], handle)

        self._vars['drt'] = dr
        self._vars['dou'] = dou_n
        self._vars['z'] = new_z


def gate_diff(x):
    return x * (-x + 1.)


def activation_diff(x):
    return (1.0 - x ** 2)


class lstm_backward_cpu(lstm_backward):

    def perform(self):
        fwd = self._fwd_op._vars
        dy = 0
        n, m = self._fwd_op._outputs.shape
        type = self._inputs[-1]['y']['cpu'].dtype
        drt = np.zeros((n, m * 4), dtype=type)
        dou = np.zeros((n, m), dtype=type)
        for grad in self._inputs:
            if 'pfgate' in grad:
                dy += grad['z']
                drt = grad['drt']
                dou = grad['dou']
            else:
                dy += grad['y']['cpu']

        #n, m = dy.shape
        dx, dw, dwr = (0, 0, 0)

        w = fwd['w']['cpu']
        wr = fwd['wr']['cpu']

        u = fwd['u']
        s = np.tanh(fwd['s'])
        x = fwd['x']
        y = fwd['y']['cpu']

        gated = fwd['gate']
        gd = gate_diff(gated)
        ps = fwd['ps']

        pfg = fwd['pfgate']
        if pfg is None:
            pfg = np.zeros((x.shape[0], w.shape[1] // 4))

        e = dy

        do = e * s * gd[:, 2 * m:]
        dou = e * gated[:, 2 * m:] * activation_diff(s) + pfg * dou

        df = dou * gd[:, :m] * ps
        di = dou * gd[:, m:2 * m] * u
        dc = dou * activation_diff(u) * gated[:, m:2 * m]

        dr = np.hstack((dc, df, di, do))
        dx += np.dot(dr, w.T)

        dw += np.dot(x.T, dr)

        dwr += np.dot(y.T, drt)

        dy = np.dot(dr, wr.T)
        drt = dr

        # Calculate ret
        self._outputs['cpu'] = dx
        self._w_out['cpu'] = dw
        self._w_r_out['cpu'] = dwr
        self._vars['z'] = dy
        self._vars['drt'] = drt
        self._vars['dou'] = dou


class LstmElement(UserGraph):

    def __init__(self, output_size, initializer=None, previous_elements=None):
        args = (output_size, initializer)
        fwd_op = lstm_forward(*args) if rm.is_cuda_active() else lstm_forward_cpu(*args)
        bwd_ops = [lstm_backward(fwd_op) if rm.is_cuda_active() else lstm_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)

    def _connect_back(self, previous_element, pos=0):
        if len(self._bwd_graphs) == 0:
            return
        pos = 0
        backward_graph_input = previous_element._get_backward_output(pos)
        if backward_graph_input is not None:
            for graph in self._bwd_graphs:
                graph.add_input(backward_graph_input)


@populate_graph
class Lstm(GraphFactory):
    '''Long short term memory [lstm]_.
    Lstm object has 8 weights and 4 biases parameters to learn.

    Weights applied to the input of the input gate, forget gate and output gate.
    :math:`W_{ij}, Wgi_{ij}, Wgf_{ij}, Wgo_{ij}`

    Weights applied to the recuurent input of the input gate, forget gate and output gate.
    :math:`R_{ij}, Rgi_{ij}, Rgf_{ij}, Rgo_{ij}`

    .. math::
        u^t_{i} &= \sum_{j = 0}^{J-1} W_{ij}x^t_{j} +
            \sum_{k = 0}^{K-1} R_{ik}y^{t-1}_{k} + b_i \\\\
        gi^t_{i} &= \sum_{j = 0}^{J-1} Wgi_{ij}x^t_{j} +
                \sum_{k = 0}^{K-1} Rgi_{ik}y^{t-1}_{k} + bi_i \\\\
        gf^t_{i} &= \sum_{j = 0}^{J-1} Wgfi_{ij}x^t_{j} +
                \sum_{k = 0}^{K-1} Rgf_{ik}y^{t-1}_{k} + bi_f \\\\
        go^t_{i} &= \sum_{j = 0}^{J-1} Wgo_{ij}x^t_{j} +
                \sum_{k = 0}^{K-1} Rgo_{ik}y^{t-1}_{k} + bi_o \\\\
        s^t_i &= sigmoid(gi^t_{i})tanh(u^t_{i}) + s^{t-1}_isigmoid(gf^t_{i}) \\\\
        y^t_{i} &= go^t_{i}tanh(s^t_{i})

    If the argument ``input_size`` is passed, this layers' weight is initialized
    in the __init__ function.
    Otherwise, the weight is initialized in its first forward calculation.


    Args:
        output_size (int): Output unit size.
        initializer (Initializer): Initializer object for weight initialization.
        weight_decay (float): Weight decay ratio.
        ignore_bias (bool): If True is given, bias will not be added.


    Example:
        >>> import numpy as np
        >>> import renom.graph as rmg
        >>> 
        >>> x = np.arange(2*3).reshape(2, 3)
        >>> layer1 = rmg.Lstm()
        >>> print(layer1(x))
        >>> 
        LSTM (F):
        [[-0.08015051, -0.07165769,  0.322052  ],
        [-0.00159989, -0.00164642,  0.5016    ]]


    .. [lstm] Felix A. Gers, Nicol N. Schraudolph, Jürgen Schmidhuber.
        Learning Precise Timing with LSTM Recurrent Networks. JMLR, 2002.
    '''

    def prepare(self, output_size=3, initializer=None, weight_decay=None, ignore_bias=False):
        self._output_size = output_size
        self._init = initializer
        self.params['w'] = graph_variable(weight_decay=weight_decay)
        self.params['wr'] = graph_variable(weight_decay=weight_decay)
        self._prev = None
        self._prevlist = []

    def reset(self):
        self._prev = None
        for p in self._prevlist:
            p.detach()
        self._prevlist = []

    def connect(self, other):
        prevs = [other, self.params['w'], self.params['wr']]
        if self._prev is not None:
            prevs.append(self._prev)
        ret = LstmElement(self._output_size, self._init, previous_elements=prevs)
        self._prevlist.append(ret)
        self._prev = ret
        return ret
