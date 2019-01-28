import renom as rm
from renom.graph.core import UserGraph, operation, GraphFactory, GraphMultiStorage, graph_variable
import renom.utility.initializer as init
import numpy as np

BATCH_NORMALIZE_FEATUREMAP = 1
BATCH_NORMALIZE_ELEMENTWISE = 2
mode_dict = {
    "activation": BATCH_NORMALIZE_ELEMENTWISE,
    "feature": BATCH_NORMALIZE_FEATUREMAP}

class batch_norm_forward(operation):

    name = 'Batch Normalize (F)'
    roles = ['inference']

    def __init__(self, momentum=0.99, epsilon=1e-5, mode=BATCH_NORMALIZE_ELEMENTWISE):
        self._momentum = momentum
        self._mode = mode
        self._epsilon = epsilon
        self._inference = False

    def setup(self, inputs):
        bias = inputs[2]['y']
        weights = inputs[1]['y']
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus

        in_shape = inputs.shape
        weight_shape = tuple([1, ] + list(in_shape[1:]))
        if self._mode == BATCH_NORMALIZE_FEATUREMAP and len(in_shape[1:]) > 2:
            weight_shape[2] = 1
            weight_shape[3] = 1
        bias_shape = weight_shape

        weights.__init__(shape=weight_shape, gpus=gpus, initializer=init.GlorotNormal())
        bias.__init__(shape=bias_shape, gpus=gpus, initializer=init.Constant(0))
        outs = GraphMultiStorage(shape=in_shape, gpus=gpus)
        mean = GraphMultiStorage(shape=weight_shape, gpus=gpus)
        sq_var = GraphMultiStorage(shape=weight_shape, gpus=gpus)
        mv_m = GraphMultiStorage(shape=weight_shape, gpus=gpus, initializer=init.Constant(0))
        mv_v = GraphMultiStorage(shape=weight_shape, gpus=gpus, initializer=init.Constant(0))

        self._inputs = inputs
        self._weights = weights
        self._bias = bias
        self._outputs = outs
        self._mean = mean
        self._sq_var = sq_var
        self._mv_m = mv_m
        self._mv_v = mv_v
        self._vars = {'y': outs, 'w': weights, 'b': bias}

    def perform(self):
        if self._mode == BATCH_NORMALIZE_ELEMENTWISE:
            axs = 0
        else:
            axs = 1
        self._axs = axs
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cuBatchNormalizatoinForward(handle, self._inputs[gpu], self._mv_m[gpu],
                                                self._mv_v[gpu], self._weights[gpu], self._bias[gpu],
                                                self._outputs[gpu], self._mean[gpu], self._sq_var[gpu],
                                                self._momentum, axs, self._inference, self._epsilon)


class batch_norm_forward_cpu(batch_norm_forward):

    def perform(self):
        if self._mode == BATCH_NORMALIZE_ELEMENTWISE:
            axs = (0,)
        else:
            axs = (0, 2, 3)
        x = self._inputs['cpu']
        w = self._weights['cpu']
        b = self._bias['cpu']
        epsilon = self._epsilon

        if self._inference:
            mean = self._mv_m['cpu']
            var = self._mv_v['cpu']
        else:
            mean = np.mean(x, axis=axs, keepdims=True)
            var = np.var(x, axis=axs, keepdims=True)

        sq_var = 1.0 / np.sqrt(var + epsilon)
        xh = (x - mean) * sq_var
        z = w * xh
        ret = z + b
        self._sq_var['cpu'] = sq_var
        self._mean['cpu'] = mean
        self._outputs['cpu'] = ret
        self._axs = axs
        if not self._inference:
            momentum = self._momentum
            N = np.prod([x.shape[s] for s in axs])
            self._mv_m['cpu'] = (1 - momentum) * self._mv_m['cpu'] + \
                                momentum * mean
            self._mv_v['cpu'] = (1 - momentum) * self._mv_v['cpu'] + \
                                momentum * var * N / max(N - 1., 1.)


class batch_norm_backward(operation):

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus

        self._inputs = inputs
        self._fwd_ins = self._fwd_op._inputs
        self._fwd_w = self._fwd_op._weights
        self._mean = self._fwd_op._mean
        self._var = self._fwd_op._sq_var
        self._outputs = GraphMultiStorage(
            shape=inputs.shape, gpus=self.gpus, initializer=init.Constant(0))
        self._weights_back = GraphMultiStorage(
            shape=self._fwd_w.shape, gpus=self.gpus, initializer=init.Constant(1))
        self._bias_back = GraphMultiStorage(
            shape=self._fwd_op._bias.shape, gpus=self.gpus, initializer=init.Constant(1))
        self._vars = {'y': self._outputs, 'dy': self._outputs, 'w': self._weights_back, 'b': self._bias_back, id(
            self._fwd_w): self._weights_back, id(self._fwd_ins): self._outputs, id(self._fwd_op._bias): self._bias_back}

    def perform(self):
        axs = self._fwd_op._axs
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cuBatchNormalizatoinBackward(handle, self._fwd_ins[gpu], self._fwd_w[gpu], self._inputs[gpu],
                                                 self._mean[gpu], self._var[gpu], self._outputs[gpu],
                                                 self._weights_back[gpu], self._bias_back[gpu], axs)


class batch_norm_backward_cpu(batch_norm_backward):

    def perform(self):
        axs = self._fwd_op._axs
        sq_var = self._var['cpu']
        mean = self._mean['cpu']
        x = self._fwd_ins['cpu']
        w = self._fwd_w['cpu']
        dy = self._inputs['cpu']

        meaned = x - mean
        N = np.prod([x.shape[s] for s in axs])

        dxh = dy * w
        ds = np.sum(dxh * meaned * -np.power(sq_var, 3) / 2, axis=axs, keepdims=True)
        du = np.sum(-dxh * sq_var, axis=axs, keepdims=True)
        dx = dxh * sq_var + (ds * 2 * meaned + du) / N
        self._outputs['cpu'] = dx

        xh = meaned * sq_var
        dw = np.sum(xh * dy, axis=axs, keepdims=True)
        self._weights_back['cpu'] = dw

        db = np.sum(dy, axis=axs, keepdims=True)
        self._bias_back['cpu'] = db


class BatchNormalizer(UserGraph):

    def __init__(self, momentum=0.99, epsilon=1e-5, mode='activation', previous_elements=None):
        fwd_op = batch_norm_forward() if rm.is_cuda_active() else batch_norm_forward_cpu()
        bwd_ops = [batch_norm_backward(fwd_op) if rm.is_cuda_active()
                   else batch_norm_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


class BatchNormalizeGraphElement(GraphFactory):
    '''See :class:`.BatchNormalize` for more.
    '''

    def __init__(self, momentum=0.99, epsilon=1e-5, mode=BATCH_NORMALIZE_ELEMENTWISE, weight_decay=None, ignore_bias=False):
        super().__init__()
        self._mom = momentum
        self._eps = epsilon
        self._mod = mode
        self.params['w'] = graph_variable(weight_decay=weight_decay)
        self.params['b'] = graph_variable(allow_update=not ignore_bias)

    def connect(self, other):
        ret = BatchNormalizer(self._mom, self._eps, self._mod, previous_elements=[
                              other, self.params['w'], self.params['b']])
        return ret
