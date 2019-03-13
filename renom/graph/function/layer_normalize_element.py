import renom as rm
from renom.graph.core import UserGraph, operation, GraphFactory, graph_variable, GraphMultiStorage
import numpy as np
import renom.utility.initializer as init


def get_mu(x):
    _ax = tuple([r for r in range(1, len(x.shape[1:]) + 1)])
    H = np.prod(x.shape[1:])
    sum = np.sum(x, axis=_ax, keepdims=True)
    return sum / H


def get_sigma(x, mu=None):
    _ax = tuple([r for r in range(1, len(x.shape[1:]) + 1)])
    H = np.prod(x.shape[1:])
    if mu is None:
        mu = get_mu(x)
    sum = np.sum(np.power(x - mu, 2), axis=_ax, keepdims=True)
    return np.sqrt(sum / H)


def get_mu_diff(x):
    H = float(np.prod(x.shape[1:]))
    return 1 / H


def get_sigma_diff(x):
    _ax = tuple([r for r in range(1, len(x.shape[1:]) + 1)])
    H = np.prod(x.shape[1:])
    mu = get_mu(x)
    sigma = get_sigma(x, mu) + 1e-8
    inside = (2 * x + H * (2 * mu / H) - 2 * (np.sum(x, axis=_ax, keepdims=True) / H + mu)) / H
    ret = 1 / (2 * sigma) * inside
    return ret


class layer_norm_forward(operation):

    name = 'Layer Normalize (F)'

    def __init__(self, gain):
        self._g = gain

    def setup(self, inputs):
        gain = inputs[1]['y']
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus
        self._inputs = inputs

        gain_shape = (1, np.prod(inputs.shape[1:]))

        gain.__init__(shape=gain_shape, gpus=self.gpus, initializer=init.Constant(self._g))
        out_shape = inputs.shape
        outs = GraphMultiStorage(shape=out_shape, gpus=self.gpus)

        self._vars = {'y': outs, 'g': gain}
        self._outputs = outs
        self._gain = gain

    def perform(self):
        self._normalized = {}
        self._mu = {}
        self._sigma = {}
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            x = self._inputs[gpu]
            gain = self._gain[gpu]
            _ax = tuple([r for r in range(1, len(x.shape[1:]) + 1)])
            H = float(np.prod(x.shape[1:]))

            sum1 = rm.cuda.cusum(x, handle, axis=_ax, keepdims=True)
            mu = sum1 / H

            sum2 = rm.cuda.cusum((x - mu) ** 2, handle, axis=_ax, keepdims=True)
            sigma = sum2.empty_like_me()
            rm.cuda.cusqrt(sum2, sigma)
            sigma = (sigma / H) + 1e-5
            normalized = (x - mu) / sigma

            self._normalized[gpu] = normalized
            self._mu[gpu] = mu
            self._sigma[gpu] = sigma

            ret = normalized * gain
            self._outputs[gpu] = ret


class layer_norm_forward_cpu(layer_norm_forward):

    def perform(self):
        x = self._inputs['cpu']
        gain = self._gain['cpu']

        mu = get_mu(x)
        sigma = get_sigma(x, mu) + 1e-8
        normalized = (x - mu) / sigma

        self._normalized = normalized
        self._mu = mu
        self._sigma = sigma

        ret = normalized * gain
        self._outputs['cpu'] = ret


class layer_norm_backward(operation):

    name = 'Layer Normalize (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        inputs = inputs[0]['y']
        gpus = inputs.gpus
        self.gpus = gpus
        self._inputs = inputs

        outs = GraphMultiStorage(shape=self._fwd_op._inputs.shape, gpus=self.gpus)
        self._gain = self._fwd_op._gain
        self._outputs = outs
        gain_out = GraphMultiStorage(shape=self._gain.shape, gpus=self.gpus)
        self._gain_out = gain_out
        self._vars = {'y': outs, id(self._fwd_op._inputs): outs,
                      'g': gain_out, id(self._gain): gain_out}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            dy = self._inputs[gpu]
            x = self._fwd_op._inputs[gpu]
            mu = self._fwd_op._mu[gpu]
            sigma = self._fwd_op._sigma[gpu]
            gain = self._gain[gpu]
            normalized = self._fwd_op._normalized[gpu]
            _ax = tuple([r for r in range(1, len(x.shape[1:]) + 1)])

            H = float(np.prod(x.shape[1:]))
            mu_diff = 1 / H

            inside = (2 * x + H * (2 * mu / H) - 2 *
                      (rm.cuda.cusum(x, handle, axis=_ax, keepdims=True) / H + mu)) / H
            sigma_diff = 1 / (2 * sigma) * inside

            dx = dy / sigma \
                - sigma_diff * rm.cuda.cusum(x * dy, handle, axis=_ax, keepdims=True) / (sigma ** 2) \
                - rm.cuda.cusum(mu_diff * dy, handle, axis=_ax, keepdims=True) / sigma \
                + sigma_diff * rm.cuda.cusum(dy, handle, axis=_ax,
                                             keepdims=True) * mu / (sigma ** 2)
            dx *= gain
            dgain = rm.cuda.cusum(normalized * dy, handle, axis=0, keepdims=True)

            self._outputs[gpu] = dx
            self._gain_out[gpu] = dgain


class layer_norm_backward_cpu(layer_norm_backward):

    def perform(self):
        dy = self._inputs['cpu']
        x = self._fwd_op._inputs['cpu']
        mu = self._fwd_op._mu
        sigma = self._fwd_op._sigma
        gain = self._gain['cpu']
        sigma_diff = get_sigma_diff(x)
        mu_diff = get_mu_diff(x)
        normalized = self._fwd_op._normalized
        _ax = tuple([r for r in range(1, len(x.shape[1:]) + 1)])

        dx = dy / sigma \
            - sigma_diff * np.sum(x * dy, axis=_ax, keepdims=True) / np.power(sigma, 2) \
            - np.sum(mu_diff * dy, axis=_ax, keepdims=True) / sigma \
            + sigma_diff * np.sum(dy, axis=_ax, keepdims=True) * mu / np.power(sigma, 2)
        dx *= gain
        dgain = np.sum(normalized * dy, axis=0, keepdims=True)

        self._outputs['cpu'] = dx
        self._gain_out['cpu'] = dgain


class LayerNormElement(UserGraph):

    def __init__(self, gain, previous_elements=None):
        fwd_op = layer_norm_forward(gain) if rm.is_cuda_active() else layer_norm_forward_cpu(gain)
        bwd_ops = [layer_norm_backward(fwd_op) if rm.is_cuda_active()
                   else layer_norm_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_elements)


class LayerNormalize(GraphFactory):
    '''Layer Normalization Model [layer_normalize]_.
    Applies a shift to a standard bell curve formation for each input unit.
    The resultant bell curve can be transformed with the gain/bias parameters, 
    displacing the mean with the bias or the variance with gain.

    Args:
        gain (float): Initial value of gain.
        bias (float): Initial value of bias.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> x = np.random.rand(2,3)
        >>> layer = rm.LayerNormalize(bias=0)
        >>> x
        array([[0.5833913 , 0.39715111, 0.19503325],
               [0.74007066, 0.34642472, 0.57293373]])
        >>> layer(x)
        layer_normalize([[ 0.12076415,  0.00333703, -0.12410118],
                   [ 0.11587134, -0.12813905,  0.01226771]])

    .. [layer_normalize] Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton. Layer Normalization.
        arXiv:1607.06450, 2016.

    '''


    def __init__(self, gain=0.1):
        super().__init__()
        self._gain = gain
        self.params['g'] = graph_variable()

    def connect(self, other):
        ret = LayerNormElement(self._gain, previous_elements=[other, self.params['g']])
        return ret