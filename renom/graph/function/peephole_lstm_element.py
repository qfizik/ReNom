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


class peephole_lstm_forward(operation):

    name = 'Peephole LSTM (F)'
    consumes = ['w', 'wr', 'wc']

    def __init__(self, output_size, initializer=None):
        self._output_size = output_size
        self._init = init.GlorotNormal() if initializer is None else initializer

    def setup(self, inputs):
        if len(inputs) > 4:
            prev_lstm = inputs[4]
        else:
            prev_lstm = None

        weights_c = inputs[3]['y']
        weights_r = inputs[2]['y']
        weights = inputs[1]['y']
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus
        self._inputs = inputs
        input_shape = inputs.shape

        weight_shape = (input_shape[1], self._output_size * 4)
        weight_r_shape = (self._output_size, self._output_size * 4)
        weight_c_shape = (1, self._output_size * 3)
        out_shape = (input_shape[0], self._output_size)
        weights.__init__(shape=weight_shape, gpus=self.gpus, initializer=self._init)
        weights_r.__init__(shape=weight_r_shape, gpus=self.gpus, initializer=self._init)
        weights_c.__init__(shape=weight_c_shape, gpus=self.gpus, initializer=self._init)
        outs = GraphMultiStorage(shape=out_shape, gpus=self.gpus)
        self._vars = {'y': outs, 'w': weights, 'wr': weights_r, 'wc': weights_c, 'pfgate': None}
        self._weights = weights
        self._weights_r = weights_r
        self._weights_c = weights_c
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
            wc = self._weights_c[gpu]

            tmp1 = rm.GPUValue(shape=(x.shape[0], w.shape[1]))
            tmp2 = rm.GPUValue(shape=(x.shape[0], w.shape[1]))

            rm.cuda.cublas_gemm(x, 0, w, 0, tmp1, handle)
            rm.cuda.cublas_gemm(pz[gpu], 0, wr, 0, tmp2, handle)

            #u = tmp1 + tmp2
            rm.cuda.cuadd(tmp1, tmp2, new_u[gpu], handle)
            rm.cuda.cupeepholelstm_forward(new_u[gpu], wc, ps[gpu], new_s[gpu], new_z[gpu])

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


class peephole_lstm_forward_cpu(peephole_lstm_forward):

    def perform(self):
        prev = self._prev
        x = self._inputs['cpu']
        w = self._weights['cpu']
        wr = self._weights_r['cpu']
        wc = self._weights_c['cpu']
        if prev is not None:
            s = prev['s']
            z = prev['z']
        else:
            s = np.zeros((x.shape[0], w.shape[1] // 4), dtype=rm.precision)
            z = np.zeros((x.shape[0], w.shape[1] // 4), dtype=rm.precision)

        u = np.dot(x, w) + np.dot(z, wr)
        m = u.shape[1] // 4
        u, gate_u = np.split(u, [m, ], axis=1)
        u = np.tanh(u)

        fg = sigmoid(s * wc[:, :m] + gate_u[:, :m])
        ig = sigmoid(s * wc[:, m:2 * m] + gate_u[:, m:2 * m])
        state = ig * u + fg * s
        og = sigmoid(state * wc[:, 2 * m:] + gate_u[:, 2 * m:])
        z = np.tanh(state) * og

        gated = np.hstack((fg, ig, og))

        ret = z
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


class peephole_lstm_backward(operation):

    name = 'Peephole LSTM (B)'
    produces = ['w', 'wr', 'wc']

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
        w_c_out = GraphMultiStorage(shape=self._fwd_op._weights_c.shape, gpus=self.gpus)
        self._outputs = outs
        self._w_out = w_out
        self._w_r_out = w_r_out
        self._w_c_out = w_c_out
        self._vars = {'y': outs, id(self._fwd_op._inputs): outs,
                      'w': w_out, id(self._fwd_op._weights): w_out,
                      'wr': w_r_out, id(self._fwd_op._weights_r): w_r_out,
                      'wc': w_c_out, id(self._fwd_op._weights_c): w_c_out,
                      'pfgate': None
                      }

    def perform(self):
        fwd = self._fwd_op._vars
        drt = None
        dou = None

        w = fwd['w']
        wr = fwd['wr']
        wc = fwd['wc']

        u = fwd['u']
        s = fwd['s']
        ps = fwd['ps']

        n = fwd['y'].shape[0]
        m = w.shape[1] // 4
        dwc = GraphMultiStorage(shape=(n, m * 3), gpus=self.gpus, initializer=init.Constant(0))

        pfg = fwd['pfgate']
        if pfg is None:
            pfg = GraphMultiStorage(shape=u.shape, gpus=self.gpus, initializer=init.Constant(0))

        for grad in self._inputs:
            if 'pfgate' in grad:
                drt = grad['drt']
                dot = grad['dou']
        if drt is None:
            drt = GraphMultiStorage(shape=u.shape, gpus=self.gpus, initializer=init.Constant(0))
            dot = GraphMultiStorage(shape=fwd['y'].shape,
                                    gpus=self.gpus, initializer=init.Constant(0))

        dr = GraphMultiStorage(shape=u.shape, gpus=self.gpus)
        dou = GraphMultiStorage(shape=fwd['y'].shape, gpus=self.gpus)
        new_z = GraphMultiStorage(shape=fwd['y'].shape, gpus=self.gpus)
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            dy = fwd['y'][gpu].zeros_like_me()
            for grad in self._inputs:
                if 'pfgate' in grad:
                    dy += grad['z'][gpu]
                else:
                    dy += grad['y'][gpu]

            rm.cuda.cupeepholelstm_backward(u[gpu], ps[gpu], s[gpu], pfg[gpu],
                                            wc[gpu], dy, drt[gpu], dot[gpu], dr[gpu], dou[gpu], dwc[gpu])
            # dx
            rm.cuda.cublas_gemm(dr[gpu], 0, w[gpu], 1, self._outputs[gpu], handle)

            # dw
            rm.cuda.cublas_gemm(fwd['x'][gpu], 1, dr[gpu], 0, self._w_out[gpu], handle)

            # dwc
            d_wc = rm.cuda.cusum(dwc[gpu], handle, axis=0, keepdims=True)
            self._w_c_out[gpu] = d_wc

            # dwr
            rm.cuda.cublas_gemm(fwd['y'][gpu], 1, drt[gpu], 0, self._w_r_out[gpu], handle)

            # dz
            rm.cuda.cublas_gemm(dr[gpu], 0, wr[gpu], 1, new_z[gpu], handle)

        self._vars['drt'] = dr
        self._vars['dou'] = dou
        self._vars['z'] = new_z


def gate_diff(x):
    return x * (-x + 1.)


def activation_diff(x):
    return (1.0 - x ** 2)


class peephole_lstm_backward_cpu(peephole_lstm_backward):

    def perform(self):
        fwd = self._fwd_op._vars
        dy = 0
        n, m = self._fwd_op._outputs.shape
        type = self._inputs[-1]['y']['cpu'].dtype
        drt = np.zeros((n, m * 4), dtype=type)
        dot = np.zeros((n, m), dtype=type)
        for grad in self._inputs:
            if 'pfgate' in grad:
                dy += grad['z']
                drt = grad['drt']
                dot = grad['dot']
            else:
                dy += grad['y']['cpu']

        #n, m = dy.shape
        dx, dw, dwr = (0, 0, 0)

        w = fwd['w']['cpu']
        wr = fwd['wr']['cpu']
        wc = fwd['wc']['cpu']

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

        # Start Fix

        do = dy * s * gd[:, 2 * m:]
        dou = dy * gated[:, 2 * m:] * activation_diff(s) + do * wc[:, 2 * m:]

        dou += pfg * dot + drt[:, m:2 * m] * wc[:, :m] + drt[:, 2 * m:3 * m] * wc[:, m:2 * m]

        df = dou * gd[:, :m] * ps if ps is not None else np.zeros_like(dou)
        di = dou * gd[:, m:2 * m] * u
        du = dou * activation_diff(u) * gated[:, m:2 * m]

        dr = np.hstack((du, df, di, do))

        dx = np.dot(dr, w.T)

        dw = np.dot(x.T, dr)

        dwr = np.dot(y.T, drt)

        dwc = np.zeros(wc.shape, dtype=wc.dtype)
        dwc[:, 2 * m:] = np.sum(do * fwd['s'], axis=0)
        dwc[:, :m] = np.sum(drt[:, m:2 * m] * fwd['s'], axis=0)
        dwc[:, m:2 * m] = np.sum(drt[:, 2 * m:3 * m] * fwd['s'], axis=0)

        dy = np.dot(dr, wr.T)

        # End fix

        # Calculate ret
        self._outputs['cpu'] = dx
        self._w_out['cpu'] = dw
        self._w_r_out['cpu'] = dwr
        self._w_c_out['cpu'] = dwc
        self._vars['z'] = dy
        self._vars['drt'] = dr
        self._vars['dot'] = dou


class PeepholeLstmElement(UserGraph):

    def __init__(self, output_size, initializer=None, previous_elements=None):
        args = (output_size, initializer)
        fwd_op = peephole_lstm_forward(
            *args) if rm.is_cuda_active() else peephole_lstm_forward_cpu(*args)
        bwd_ops = [peephole_lstm_backward(fwd_op) if rm.is_cuda_active()
                   else peephole_lstm_backward_cpu(fwd_op)]
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
class PeepholeLstm(GraphFactory):

    def prepare(self, output_size=3, initializer=None, weight_decay=None, ignore_bias=False):
        self._output_size = output_size
        self._init = initializer
        self.params['w'] = graph_variable(weight_decay=weight_decay)
        self.params['wr'] = graph_variable(weight_decay=weight_decay)
        self.params['wc'] = graph_variable(weight_decay=weight_decay)
        self._prev = None
        self._prevlist = []

    def reset(self):
        self._prev = None
        for p in self._prevlist:
            p.detach()
        self._prevlist = []

    def connect(self, other):
        prevs = [other, self.params['w'], self.params['wr'], self.params['wc']]
        if self._prev is not None:
            prevs.append(self._prev)
        ret = PeepholeLstmElement(self._output_size, self._init, previous_elements=prevs)
        self._prevlist.append(ret)
        self._prev = ret
        return ret
