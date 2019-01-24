import renom as rm
from renom.graph.core import UserGraph, operation, GraphFactory, graph_variable, GraphMultiStorage, StateHolder
import numpy as np
import renom.utility.initializer as init


class gru_forward(operation):

    name = 'Gru (F)'

    def __init__(self, output_size):
        self._output_size = output_size

    def setup(self, inputs):
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
        it = init.GlorotNormal()
        #it = init.Constant(1)
        weights.__init__(shape=weight_shape, gpus=self.gpus, initializer=it)
        weights_r.__init__(shape=weight_r_shape, gpus=self.gpus, initializer=it)
        bias.__init__(shape=(1, self._output_size * 3),
                      gpus=self.gpus, initializer=init.Constant(1))
        outs = GraphMultiStorage(shape=out_shape, gpus=self.gpus)
        self._vars = {'y': outs, 'w': weights, 'wr': weights_r, 'b': bias}
        self._weights = weights
        self._weights_r = weights_r
        self._bias = bias
        self._outputs = outs
        self._state = None

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            x = self._inputs[gpu]
            w = self._weights[gpu]
            u = self._weights_r[gpu]

            m = w.shape[1] // 3
            if self._state is None:
                shp = (x.shape[0], m)
                self._state = StateHolder()
                for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
                    self._state.set_prev('z' + str(gpu), rm.GPUValue(np.zeros(shp)))

            hminus = self._state.get_prev('z' + str(gpu))
            # Perform Forward Calcuations
            dotted = rm.GPUValue(shape=(x.shape[0], w.shape[1]))
            rm.cuda.cublas_gemm(x, 0, w, 0, dotted, handle)
            ABC = dotted.empty_like_me()
            h = hminus.empty_like_me()
            rm.cuda.cugru_forward(dotted, hminus, u, ABC, h)

            self._outputs[gpu] = h
            self._state.push({'x' + str(gpu): x,
                              'z' + str(gpu): h,
                              'y' + str(gpu): h,
                              'hminus' + str(gpu): hminus,
                              'ABC' + str(gpu): ABC,
                              })

    def reset(self):
        self._state = None


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class gru_forward_cpu(gru_forward):

    def perform(self):
        x = self._inputs['cpu']
        w = self._weights['cpu']
        u = self._weights_r['cpu']
        b = self._bias['cpu']
        if self._state is None:
            self._state = StateHolder({
                'z': np.zeros((x.shape[0], w.shape[1] // 3), dtype=rm.precision),
            })
        pz = self._state.get_prev('z')

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
        self._state.push({
            'x': x,
            'z': h,
            'A': A,
            'B': B,
            'C': C,
            'hminus': pz,
        })


class gru_backward(operation):

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus
        self._inputs = inputs
        self._state = None
        outs = GraphMultiStorage(shape=self._fwd_op._inputs.shape, gpus=self.gpus)
        w_out = GraphMultiStorage(shape=self._fwd_op._weights.shape, gpus=self.gpus)
        w_r_out = GraphMultiStorage(shape=self._fwd_op._weights_r.shape, gpus=self.gpus)
        self._outputs = outs
        self._w_out = w_out
        self._w_r_out = w_r_out
        self._vars = {'y': outs, id(self._fwd_op._inputs): outs,
                      'w': w_out, id(self._fwd_op._weights): w_out,
                      'wr': w_r_out, id(self._fwd_op._weights_r): w_r_out,
                      }

    def perform(self):
        self._state = self._fwd_op._state
        _t = self._state._cur_time
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            dy = self._inputs[gpu]
            w = self._fwd_op._weights[gpu]
            u = self._fwd_op._weights_r[gpu]

            self._outputs[gpu] = self._outputs[gpu].zeros_like_me()
            self._w_out[gpu] = self._w_out[gpu].zeros_like_me()
            self._w_r_out[gpu] = self._w_r_out[gpu].zeros_like_me()

            self._state._cur_time = _t
            while(self._state._cur_time > 0):
                cur_state = self._state.peek()
                x = cur_state['x' + str(gpu)]
                #y = cur_state['y' + str(gpu)]
                hminus = cur_state['hminus' + str(gpu)]
                ABC = cur_state['ABC' + str(gpu)]

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

                self._outputs[gpu] += dx
                self._w_out[gpu] += dw
                self._w_r_out[gpu] += du

                self._state._cur_time -= 1
                dy = dpz


def sigmoid_diff(x):
    return sigmoid(x) * (-sigmoid(x) + 1.)


def tanh_diff(x):
    return (1.0 - np.tanh(x) ** 2)


class gru_backward_cpu(gru_backward):

    def perform(self):
        self._state = self._fwd_op._state

        dy = self._inputs['cpu']
        n, m = dy.shape
        dx, dw, dwr = (0, 0, 0)

        w = self._fwd_op._weights['cpu']
        u = self._fwd_op._weights_r['cpu']
        w_z, w_r, w_h = np.split(w, [m, m * 2, ], axis=1)
        u_z, u_r, u_h = np.split(u, [m, m * 2], axis=1)

        while(self._state._cur_time > 0):
            cur_state = self._state.peek()
            x = cur_state['x']
            A = cur_state['A']
            B = cur_state['B']
            C = cur_state['C']
            hminus = cur_state['hminus']
            y = dy

            dA = sigmoid_diff(A)
            dB = sigmoid_diff(B)
            dC = tanh_diff(C)

            # Calculate dx
            dx_z = np.dot(y * dA, w_z.T)
            dx_r = np.dot(y * dB * dC * u_h * hminus, w_r.T)
            dx_h = np.dot(y * dC, w_h.T)
            dx += dx_z + dx_r + dx_h

            # Calculate dw
            dw_z = np.dot(x.T, y * dA)
            dw_r = np.dot(x.T, y * dB * dC * u_h * hminus)
            dw_h = np.dot(x.T, y * dC)
            dw += np.concatenate([dw_z, dw_r, dw_h], axis=1)

            du_z = np.sum(dA * hminus * y, axis=0, keepdims=True)
            du_r = np.sum(y * dC * dB * u_h * hminus * hminus, axis=0, keepdims=True)
            du_h = np.sum(sigmoid(B) * dC * y * hminus, axis=0, keepdims=True)
            dwr += np.concatenate([du_z, du_r, du_h], axis=1)
            pz_z = y * dA * u_z
            pz_r = y * dC * dB * u_h * hminus * u_r
            pz_h = y * dC * sigmoid(B) * u_h

            dy = pz_z + pz_r + pz_h

            self._state.pop()

        # Calculate ret
        self._outputs['cpu'] = dx
        self._w_out['cpu'] = dw
        self._w_r_out['cpu'] = dwr


class GruElement(UserGraph):

    def __init__(self, output_size, previous_elements=None):
        fwd_op = gru_forward(output_size) if rm.is_cuda_active() else gru_forward_cpu(output_size)
        bwd_ops = [gru_backward(fwd_op) if rm.is_cuda_active() else gru_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


class GruGraphElement(GraphFactory):

    def __init__(self, output_size=3, weight_decay=None):
        super().__init__()
        self._output_size = output_size
        self.params['w'] = graph_variable(weight_decay=weight_decay)
        self.params['wr'] = graph_variable(weight_decay=weight_decay)
        self.params['b'] = graph_variable()

    def connect(self, other):
        ret = GruElement(self._output_size, previous_elements=[
                         other, self.params['w'], self.params['wr'], self.params['b']])
        return ret
