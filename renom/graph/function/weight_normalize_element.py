import renom as rm
from renom.graph.core import UserGraph, operation, GraphFactory, graph_variable, GraphMultiStorage
import numpy as np
import renom.utility.initializer as init


class weight_norm_forward(operation):

    name = 'Weight Normalization (F)'

    def __init__(self, output_size, gain, initializer=None):
        self._output_size = output_size
        self._g = gain
        self._init = init.GlorotNormal() if initializer is None else initializer

    def setup(self, inputs):
        gain = inputs[2]['y']
        weights = inputs[1]['y']
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus
        self._inputs = inputs

        gain_shape = (1, self._output_size)
        weight_shape = (inputs.shape[1], self._output_size)
        out_shape = (inputs.shape[0], self._output_size)

        gain.__init__(shape=gain_shape, gpus=self.gpus, initializer=init.Constant(self._g))
        weights.__init__(shape=weight_shape, gpus=self.gpus, initializer=self._init)
        outs = GraphMultiStorage(shape=out_shape, gpus=self.gpus)

        self._vars = {'y': outs, 'w': weights, 'g': gain}
        self._outputs = outs
        self._weights = weights
        self._gain = gain

    def perform(self):
        self._norms = {}
        self._gained_ws = {}
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            x = self._inputs[gpu]
            w = self._weights[gpu]
            g = self._gain[gpu]
            norm = rm.cuda.cusum((w * w), handle, keepdims=True)
            rm.cuda.cusqrt(norm, norm)
            rm.cuda.cuadd(norm, 1e-7, norm, handle)
            self._norms[gpu] = norm
            gained_w = w / norm * g
            self._gained_ws[gpu] = gained_w
            rm.cuda.cublas_gemm(x, 0, gained_w, 0, self._outputs[gpu], handle)


class weight_norm_forward_cpu(weight_norm_forward):

    def perform(self):
        x = self._inputs['cpu']
        w = self._weights['cpu']
        g = self._gain['cpu']
        norm = np.sqrt(np.sum(w * w, keepdims=True))
        gained_w = w / norm * g
        self._g_w = gained_w
        ret = np.dot(x, gained_w)
        self._outputs['cpu'] = ret


class weight_norm_backward(operation):

    name = 'Weight Normalization (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus
        self._inputs = inputs

        outs = GraphMultiStorage(shape=self._fwd_op._inputs.shape, gpus=self.gpus)
        self._gain = self._fwd_op._gain
        self._weight = self._fwd_op._weights
        self._outputs = outs
        weights_out = GraphMultiStorage(shape=self._weight.shape, gpus=self.gpus)
        gain_out = GraphMultiStorage(shape=self._gain.shape, gpus=self.gpus)
        self._weights_out = weights_out
        self._gain_out = gain_out
        self._vars = {'y': outs, id(self._fwd_op._inputs): outs,
                      'w': weights_out, id(self._weight): weights_out,
                      'g': gain_out, id(self._gain): gain_out}

    def perform(self):
        gained_ws = self._fwd_op._gained_ws
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            x = self._fwd_op._inputs[gpu]
            w = self._weight[gpu]
            gain = self._gain[gpu]
            dy = self._inputs[gpu]
            g_w = gained_ws[gpu]
            dx = self._outputs[gpu]

            rm.cuda.cublas_gemm(dy, 0, g_w, 1, dx, handle)

            normal_dw = w.empty_like_me()
            rm.cuda.cublas_gemm(x, 1, dy, 0, normal_dw, handle)
            dw = g_w / w * (normal_dw - rm.cuda.cusum(g_w * normal_dw
                                                      / gain, handle, keepdims=True) * g_w / gain)
            dgain = rm.cuda.cusum(normal_dw * g_w / gain, handle, axis=0, keepdims=True)

            self._outputs[gpu] = dx
            self._weights_out[gpu] = dw
            self._gain_out[gpu] = dgain


class weight_norm_backward_cpu(weight_norm_backward):

    def perform(self):
        x = self._fwd_op._inputs['cpu']
        dy = self._inputs['cpu']
        g_w = self._fwd_op._g_w
        w = self._weight['cpu']
        gain = self._gain['cpu']
        dx = np.dot(dy, g_w.T)

        normal_dw = np.dot(x.T, dy)
        dw = g_w / w * (normal_dw - np.sum(g_w * normal_dw / gain, keepdims=True) * g_w / gain)
        dgain = np.sum(normal_dw * g_w / gain, axis=0, keepdims=True)

        self._outputs['cpu'] = dx
        self._weights_out['cpu'] = dw
        self._gain_out['cpu'] = dgain


class WeightNormElement(UserGraph):

    def __init__(self, output_size, gain, initializer=None, previous_elements=None):
        args = (output_size, gain, initializer)
        fwd_op = weight_norm_forward(
            *args) if rm.is_cuda_active() else weight_norm_forward_cpu(*args)
        bwd_ops = [weight_norm_backward(fwd_op) if rm.is_cuda_active()
                   else weight_norm_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


class WeightNormalize(GraphFactory):

    def __init__(self, output_size=1, gain=0.1, initializer=None, weight_decay=None, ignore_bias=None):
        # TODO: Add bias.
        super().__init__()
        self._gain = gain
        self._output_size = output_size
        self._init = initializer
        self.params['g'] = graph_variable()
        self.params['w'] = graph_variable(weight_decay=weight_decay)

    def connect(self, other):
        ret = WeightNormElement(self._output_size, self._gain, self._init, previous_elements=[
                                other, self.params['w'], self.params['g']])
        return ret
