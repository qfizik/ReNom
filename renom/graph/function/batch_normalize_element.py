import renom as rm
from renom.layers.function.batch_normalize import mode_dict
from renom.graph.core import UserGraph, operation, GraphFactory, GraphMultiStorage, graph_variable
import renom.utility.initializer as init
import numpy as np


class batch_norm_forward(operation):

    name = 'Batch Normalize (F)'
    consumes = ['w', 'b']
    roles = ['inference']

    def __init__(self, momentum=0.99, epsilon=1e-5, axis=None, initializer=None):
        self._momentum = momentum
        # For v2 load support
        if axis == mode_dict['activation']:
            axis = None
        elif axis == mode_dict['feature']:
            axis = 1
        self._axis = axis
        self._epsilon = epsilon
        self._inference = False
        self._init = init.GlorotNormal() if initializer is None else initializer

    def setup(self, inputs):
        mv_v = inputs[4]['y']
        mv_m = inputs[3]['y']
        bias = inputs[2]['y']
        weights = inputs[1]['y']
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus

        in_shape = inputs.shape
        weight_shape = [1, ] + list(in_shape[1:])
        if self._axis == 1 and len(in_shape) > 2:
            weight_shape[2] = 1
            weight_shape[3] = 1
        weight_shape = tuple(weight_shape)
        bias_shape = weight_shape

        weights.__init__(shape=weight_shape, gpus=gpus, initializer=self._init)
        bias.__init__(shape=bias_shape, gpus=gpus, initializer=init.Constant(0))
        outs = GraphMultiStorage(shape=in_shape, gpus=gpus)
        mean = GraphMultiStorage(shape=weight_shape, gpus=gpus)
        sq_var = GraphMultiStorage(shape=weight_shape, gpus=gpus)
        mv_m.__init__(shape=weight_shape, gpus=gpus, initializer=init.Constant(0))
        mv_v.__init__(shape=weight_shape, gpus=gpus, initializer=init.Constant(0))

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
        if self._axis is None:
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
        if self._axis is None:
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

    name = 'Batch Normalize (B)'
    produces = ['w', 'b']

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


class BatchNormalizeElement(UserGraph):

    def __init__(self, momentum=0.99, epsilon=1e-5, axis=None, initializer=None, previous_elements=None):
        assert axis in [None, 1], "BatchNormalizeElement accepts 1 or None as axis."
        args = (momentum, epsilon, axis, initializer)
        fwd_op = batch_norm_forward(*args) if rm.is_cuda_active() else batch_norm_forward_cpu(*args)
        bwd_ops = [batch_norm_backward(fwd_op) if rm.is_cuda_active()
                   else batch_norm_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


class BatchNormalize(GraphFactory):
    """Batch normalization function [bn]_.
    This layer accelerates learning speed with reducing internal covariate shift
    and allow us to set high learning rate.

    When the forward propagation, if the argument ``inference`` is set to False this layer
    calculates moving average of mean and variance.
    Other wise the ``inference`` is set to True, this layer uses the moving average which
    calculated in the above mode.




    Args:
        momentum (float): Momentum coefficient for the moving average.
        epsilon (float): Small number added to avoid division by zero.
        axis (int, None): 'activation'  or 'feature'.
        ignore_bias (bool): If `True` is given, bias will not be added.
        initializer (Initializer): Initializer object for weight initialization.

    Example:
        >>> import numpy as np
        >>> import renom.graph as rmg
        >>> x = np.random.rand(3, 2)
        >>> x.shape
        (3, 2)
        >>> layer = rmg.BatchNormalize(momentum=0.99)
        >>> layer(x, inference=False)
        batch_normalize([[-0.05047419,  0.00471613],
                         [-0.00887055, -0.01459344],
                         [ 0.05934474,  0.00987731]], dtype=float32)

    .. [bn] Sergey Ioffe, Christian Szegedy. Batch Normalization:
        Accelerating Deep Network Training by Reducing Internal Covariate Shift(2015)

    """

    def prepare(self, momentum=0.99, epsilon=1e-5, axis=None, initializer=None, weight_decay=None, ignore_bias=False):
        self._mom = momentum
        self.params['_epsilon'] = epsilon
        self.params['_mode'] = axis
        self._init = initializer
        self.params['w'] = graph_variable(weight_decay=weight_decay)
        self.params['b'] = graph_variable(allow_update=not ignore_bias)
        self._mov_mean = graph_variable()
        self._mov_std = graph_variable()
        self.params['_mov_mean'] = graph_variable()
        self.params['_mov_std'] = graph_variable()

    def connect(self, other):
        ret = BatchNormalizeElement(self._mom, self.params['_epsilon'], self.params['_mode'], self._init,
                                    previous_elements=[
            other, self.params['w'], self.params['b'], self.params['_mov_mean'], self.params['_mov_std']])
        return ret
