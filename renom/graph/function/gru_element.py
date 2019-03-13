import renom as rm
from renom.graph.core import UserGraph, operation, GraphFactory, graph_variable, GraphMultiStorage, StateHolder
import numpy as np
import renom.utility.initializer as init


class gru_forward(operation):

    name = 'Gru (F)'
    consumes = ['w', 'wr']

    def __init__(self, output_size, initializer=None):
        self._output_size = output_size
        self._init = init.GlorotNormal() if initializer is None else initializer

    def setup(self, inputs):
        if len(inputs) > 4:
            prev = inputs[4]
        else:
            prev = None
        bias = inputs[3]['y']
        weights_r = inputs[2]['y']
        weights = inputs[1]['y']
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus
        self._inputs = inputs
        input_shape = inputs.shape

        weight_shape = (input_shape[1], self._output_size * 3)
        weight_r_shape = (1, self._output_size * 3)
        out_shape = (input_shape[0], self._output_size)
        weights.__init__(shape=weight_shape, gpus=self.gpus, initializer=self._init)
        weights_r.__init__(shape=weight_r_shape, gpus=self.gpus, initializer=self._init)
        bias.__init__(shape=(1, self._output_size * 3),
                      gpus=self.gpus, initializer=init.Constant(1))
        outs = GraphMultiStorage(shape=out_shape, gpus=self.gpus)
        self._vars = {'y': outs, 'x': inputs, 'w': weights, 'wr': weights_r, 'b': bias}
        self._weights = weights
        self._weights_r = weights_r
        self._prev_gru = prev
        self._bias = bias
        self._outputs = outs

    def perform(self):
        n = self._inputs.shape[0]
        if self._prev_gru is None:
            m = self._weights.shape[1] // 3
            p_z = GraphMultiStorage(shape=(n, m), gpus=self.gpus, initializer=init.Constant(0))
        else:
            p_z = self._prev_gru['z']
        new_ABC = GraphMultiStorage(shape=(n, self._weights.shape[1]), gpus=self.gpus)
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            x = self._inputs[gpu]
            w = self._weights[gpu]
            u = self._weights_r[gpu]
            hminus = p_z[gpu]
            print(hminus.shape)
            print(self._outputs.shape)

            # Perform Forward Calcuations
            dotted = rm.GPUValue(shape=(x.shape[0], w.shape[1]))
            rm.cuda.cublas_gemm(x, 0, w, 0, dotted, handle)
            ABC = new_ABC[gpu]
            h = hminus.empty_like_me()
            rm.cuda.cugru_forward(dotted, hminus, u, ABC, h)

            self._outputs[gpu] = h
        self._vars['z'] = self._outputs
        self._vars['hminus'] = p_z
        self._vars['ABC'] = new_ABC


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class gru_forward_cpu(gru_forward):

    def perform(self):
        x = self._inputs['cpu']
        w = self._weights['cpu']
        u = self._weights_r['cpu']
        b = self._bias['cpu']
        if self._prev_gru is None:
            pz = np.zeros((x.shape[0], w.shape[1] // 3), dtype=rm.precision)
        else:
            pz = self._prev_gru['z']

        m = w.shape[1] // 3
        w_z, w_r, w_h = np.split(w, [m, m * 2, ], axis=1)
        u_z, u_r, u_h = np.split(u, [m, m * 2], axis=1)
        b_z, b_r, b_h = np.split(b, [m, m * 2], axis=1)
        hminus = pz

        # Perform Forward Calcuations
        A = np.dot(x, w_z) + hminus * u_z + b_z
        B = np.dot(x, w_r) + u_r * hminus + b_r
        C = np.dot(x, w_h) + sigmoid(B) * u_h * hminus + b_h

        h = sigmoid(A) + np.tanh(C)

        # Calculate ret
        self._outputs['cpu'] = h
        self._vars['z'] = h
        self._vars['A'] = A
        self._vars['B'] = B
        self._vars['C'] = C
        self._vars['hminus'] = pz


class gru_backward(operation):

    name = 'Gru (B)'
    produces = ['w', 'wr']

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        self._inputs = inputs
        gpus = inputs[0]['y'].gpus
        self.gpus = gpus
        outs = GraphMultiStorage(shape=self._fwd_op._inputs.shape, gpus=self.gpus)
        w_out = GraphMultiStorage(shape=self._fwd_op._weights.shape, gpus=self.gpus)
        w_r_out = GraphMultiStorage(shape=self._fwd_op._weights_r.shape, gpus=self.gpus)
        self._outputs = outs
        self._w_out = w_out
        self._w_r_out = w_r_out
        self._vars = {
            'y': outs, 'dy': outs, id(self._fwd_op._inputs): outs,
            'w': w_out, id(self._fwd_op._weights): w_out,
            'wr': w_r_out, id(self._fwd_op._weights_r): w_r_out,
            'gru_unit': None,
        }

    def perform(self):
        fwd = self._fwd_op._vars
        new_z = None
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            dy = fwd['y'][gpu].zeros_like_me()
            for grad in self._inputs:
                if 'gru_unit' in grad:
                    dy += grad['z'][gpu]
                else:
                    dy += grad['y'][gpu]
            w = self._fwd_op._weights[gpu]
            u = self._fwd_op._weights_r[gpu]

            self._outputs[gpu] = self._outputs[gpu].zeros_like_me()
            self._w_out[gpu] = self._w_out[gpu].zeros_like_me()
            self._w_r_out[gpu] = self._w_r_out[gpu].zeros_like_me()

            x = fwd['x'][gpu]
            #y = cur_state['y' + str(gpu)]
            hminus = fwd['hminus'][gpu]
            ABC = fwd['ABC'][gpu]

            dx = x.empty_like_me()
            db = u.empty_like_me()
            dw = w.empty_like_me()
            yconc = ABC.empty_like_me()
            du = u.empty_like_me()
            dpz = hminus.empty_like_me()
            dxx = x.empty_like_me()

            rm.cuda.cugru_backward(ABC, dy, yconc, u,
                                   hminus, db, du, dpz, dxx)
            # Calculate dx
            rm.cuda.cublas_gemm(yconc, 0, w, 1, dx, handle)
            rm.cuda.cublas_gemm(x, 1, yconc, 0, dw, handle)

            self._outputs[gpu] = dx
            self._w_out[gpu] = dw
            self._w_r_out[gpu] = du

            if new_z is None:
                new_z = GraphMultiStorage(shape=dpz.shape, gpus=self.gpus)
            new_z[gpu] = dpz
        self._vars['z'] = new_z


def sigmoid_diff(x):
    return sigmoid(x) * (-sigmoid(x) + 1.)


def tanh_diff(x):
    return (1.0 - np.tanh(x) ** 2)


class gru_backward_cpu(gru_backward):

    def perform(self):
        fwd = self._fwd_op._vars
        dy = 0
        for grad in self._inputs:
            if 'gru_unit' in grad:
                dy += grad['z']
            else:
                dy += grad['y']['cpu']
        n, m = dy.shape

        w = fwd['w']['cpu']
        u = fwd['wr']['cpu']
        w_z, w_r, w_h = np.split(w, [m, m * 2, ], axis=1)
        u_z, u_r, u_h = np.split(u, [m, m * 2], axis=1)

        x = fwd['x']['cpu']
        A = fwd['A']
        B = fwd['B']
        C = fwd['C']
        hminus = fwd['hminus']
        y = dy

        dA = sigmoid_diff(A)
        dB = sigmoid_diff(B)
        dC = tanh_diff(C)

        # Calculate dx
        dx_z = np.dot(y * dA, w_z.T)
        dx_r = np.dot(y * dB * dC * u_h * hminus, w_r.T)
        dx_h = np.dot(y * dC, w_h.T)
        dx = dx_z + dx_r + dx_h

        # Calculate dw
        dw_z = np.dot(x.T, y * dA)
        dw_r = np.dot(x.T, y * dB * dC * u_h * hminus)
        dw_h = np.dot(x.T, y * dC)
        dw = np.concatenate([dw_z, dw_r, dw_h], axis=1)

        du_z = np.sum(dA * hminus * y, axis=0, keepdims=True)
        du_r = np.sum(y * dC * dB * u_h * hminus * hminus, axis=0, keepdims=True)
        du_h = np.sum(sigmoid(B) * dC * y * hminus, axis=0, keepdims=True)
        dwr = np.concatenate([du_z, du_r, du_h], axis=1)
        pz_z = y * dA * u_z
        pz_r = y * dC * dB * u_h * hminus * u_r
        pz_h = y * dC * sigmoid(B) * u_h

        dy = pz_z + pz_r + pz_h

        # Calculate ret
        self._outputs['cpu'] = dx
        self._vars['z'] = dy
        self._w_out['cpu'] = dw
        self._w_r_out['cpu'] = dwr


class GruElement(UserGraph):

    def __init__(self, output_size, initializer=None, previous_elements=None):
        args = (output_size, initializer)
        fwd_op = gru_forward(*args) if rm.is_cuda_active() else gru_forward_cpu(*args)
        bwd_ops = [gru_backward(fwd_op) if rm.is_cuda_active() else gru_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)

    def connect_back(self, previous_element, pos=0):
        if len(self._bwd_graphs) == 0:
            return
        pos = 0
        backward_graph_input = previous_element.get_backward_output(pos)
        if backward_graph_input is not None:
            for graph in self._bwd_graphs:
                graph.add_input(backward_graph_input)


class Gru(GraphFactory):
    """Gated recurrent unit [gru]_ layer.

    Args:
        output_size (int): 
        initializer (Initializer):
        weight_decay (float):
        ignore_bias (bool):

    Example:
        >>> import numpy as np
        >>> import renom as rm


    .. [gru] Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio. 
        Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. NIPS, 2014.

    """

    def __init__(self, output_size=3, initializer=None, weight_decay=None, ignore_bias=False):
        super().__init__()
        self._output_size = output_size
        self._init = initializer
        self.params['w'] = graph_variable(weight_decay=weight_decay)
        self.params['wr'] = graph_variable(weight_decay=weight_decay)
        self.params['b'] = graph_variable(allow_update=not ignore_bias)
        self._prev = None
        self._prevlist = []

    def reset(self):
        self._prev = None
        for p in self._prevlist:
            p.detach()
        self._prevlist = []

    def connect(self, other):
        prevs = [other, self.params['w'], self.params['wr'], self.params['b']]
        if self._prev is not None:
            prevs.append(self._prev)
        ret = GruElement(self._output_size, self._init, previous_elements=prevs)
        self._prevlist.append(ret)
        self._prev = ret
        return ret
