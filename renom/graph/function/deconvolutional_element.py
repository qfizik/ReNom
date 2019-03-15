import renom as rm
from renom.layers.function.utils import im2col, col2im, colnim, imncol
from renom.graph.core import operation, UserGraph, GraphMultiStorage, GraphFactory, graph_variable
import renom.utility.initializer as init
import numpy as np


class deconv_forward(operation):

    name = 'Deconvolution (F)'
    consumes = ['w', 'b']

    def __init__(self, channel, kernel=3, padding=0, stride=1, initializer=None):
        self._channels = channel
        self._k = kernel
        self._p = padding
        self._s = stride
        self._d = 1
        self._init = init.GlorotNormal() if initializer is None else initializer

    def setup(self, inputs):

        weights = inputs[1]['y']
        bias = inputs[2]['y']
        inputs = inputs[0]['y']
        input_shape = inputs.shape
        dims = len(input_shape[2:])
        self._dims = dims

        self._kernel = np.array(list(self._k for i in range(dims))).astype(np.int32)
        self._padding = np.array(list(self._p for i in range(dims))).astype(np.int32)
        self._stride = np.array(list(self._s for i in range(dims))).astype(np.int32)
        self._dilation = np.array(list(self._d for i in range(dims))).astype(np.int32)

        self._inputs = inputs
        gpus = inputs.gpus
        self.gpus = gpus

        weight_shape = (input_shape[1], self._channels, *self._kernel)
        bias_shape = (1, self._channels, *(1 for i in range(dims)))

        weights.__init__(shape=weight_shape, gpus=gpus, initializer=self._init)
        bias.__init__(shape=bias_shape, gpus=gpus, initializer=init.Constant(0))

        self._weights = weights
        self._bias = bias

        imgs = tuple(self._stride[i] * (input_shape[i + 2] - 1) +
                     self._kernel[i] - 2 * self._padding[i] for i in range(dims))
        output_shape = [input_shape[0], self._channels, *imgs]
        self._outputs = GraphMultiStorage(shape=output_shape, gpus=gpus)
        self._vars = {'w': self._weights, 'b': self._bias, 'y': self._outputs}

        if rm.is_cuda_active():
            with rm.cuda.RenomHandler() as handle:
                if dims == 2:
                    self._conv_desc = rm.cuda.ConvolutionDescriptor(
                        self._padding, self._stride, self._dilation, rm.precision)
                    self._filter_desc = rm.cuda.FilterDescriptor(weight_shape, rm.precision)
                else:
                    self._conv_desc = rm.cuda.ConvolutionNDescriptor(
                        self._padding, self._stride, rm.precision)
                    self._filter_desc = rm.cuda.NdFilterDescriptor(weight_shape, rm.precision)
                self._algo = rm.cuda.cuGetConvolutionFwdAlgo(
                    handle, self._conv_desc, self._filter_desc, inputs[0], self._outputs[0])
                self._bwd_algo = rm.cuda.cuGetConvolutionBwdAlgo(
                    handle, self._conv_desc, self._filter_desc, inputs[0], self._outputs[0])

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cuConvolutionBackwardData(
                handle, self._conv_desc, self._filter_desc, self._weights[gpu], self._inputs[gpu], self._outputs[gpu])
            rm.cuda.cu_add_bias(self._bias[gpu], self._outputs[gpu])


class deconv_forward_cpu(deconv_forward):

    def perform(self):
        x = self._inputs['cpu']
        w = self._weights['cpu']
        b = self._bias['cpu']

        w_rev = np.reshape(w, (w.shape[0], w.shape[1], -1))
        w_rev = np.flip(w_rev, 2).reshape(w.shape)
        col = colnim(x, w_rev, self._stride)
        col += b
        z = col
        self._outputs['cpu'] = z


class deconv_backward(operation):

    name = 'Deconvolution (B)'
    produces = ['w', 'b']

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):

        inputs = inputs[0]['y']
        self._inputs = inputs
        self._fwd_w = self._fwd_op._weights
        self._fwd_b = self._fwd_op._bias
        self._fwd_in = self._fwd_op._inputs
        self.gpus = inputs.gpus

        self._outputs = GraphMultiStorage(shape=self._fwd_in.shape, gpus=self.gpus)
        self._bias_out = GraphMultiStorage(shape=self._fwd_b.shape, gpus=self.gpus)
        self._weights_out = GraphMultiStorage(shape=self._fwd_w.shape, gpus=self.gpus)

        self._vars = {'w': self._weights_out, 'b': self._bias_out, 'y': self._outputs,
                      'dy': self._outputs,
                      id(self._fwd_in): self._outputs,
                      id(self._fwd_w): self._weights_out,
                      id(self._fwd_b): self._bias_out,
                      }

        if rm.is_cuda_active():
            self._algo = self._fwd_op._bwd_algo

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cuConvolutionForward(handle, self._fwd_op._conv_desc, self._fwd_op._filter_desc,
                                         self._inputs[gpu], self._fwd_w[gpu], self._outputs[gpu], 0)
            rm.cuda.cuConvolutionBackwardFilter(handle, self._fwd_op._conv_desc, self._fwd_op._filter_desc,
                                                self._inputs[gpu], self._fwd_in[gpu], self._weights_out[gpu])
            rm.cuda.cuConvolutionBackwardBias(handle, self._inputs[gpu], self._bias_out[gpu])


class deconv_backward_cpu(deconv_backward):

    def perform(self):
        dy = self._inputs['cpu']
        x = self._fwd_in['cpu']
        w = self._fwd_w['cpu']

        dx = imncol(dy, w, self._fwd_op._stride, padding=self._fwd_op._padding)

        l = [x for x in range(len(dy.shape))]  # noqa
        del(l[1])
        dw = np.ones_like(w) * \
            np.swapaxes(np.sum(x, axis=tuple(l), keepdims=True), 0, 1)

        db = np.sum(np.ones_like(dy), axis=tuple(
            [x for x in range(2, len(dy.shape), 1)]), keepdims=True)
        db = np.sum(db, axis=0, keepdims=True)

        self._outputs['cpu'] = dx
        self._weights_out['cpu'] = dw
        self._bias_out['cpu'] = db


class DeconvolutionalGraph(UserGraph):

    def __init__(self, channel=3, kernel=3, padding=0, stride=1, initializer=None, previous_element=None):
        args = (channel, kernel, padding, stride, initializer)
        fwd_op = deconv_forward(*args) if rm.is_cuda_active() else deconv_forward_cpu(*args)
        bwd_ops = [deconv_backward(fwd_op) if rm.is_cuda_active()
                   else deconv_backward_cpu(fwd_op)]

        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_element)


class Deconv(GraphFactory):
    """Deconvolutional Layer.

      This class creates a deconvolution filter to be convolved with
      the input tensor. This class accepts up to 3d image input.
      Note that the 2d implementation differs slightly from the 3d implementation, giving no
      guarantee that they will perform equally.

      Args:
          channel (int, tuple): The dimensionality of the output.
          kernel (int, tuple): Filter size of the convolution kernel.
          padding (int, tuple): Size of the zero-padding around the image.
          stride (int, tuple): Stride-size of the convolution.
          initializer (Initializer): Initializer object for weight initialization.
          weight_decay (float): Weight decay ratio. This must be None or 0 <= ratio.
          ignore_bias (bool): If True is given, bias term will be ignored.

      Example:
          >>> import numpy as np
          >>> import renom.graph as rmg

      Note:
          Tensor data format is **NCHW***.
    """

    def prepare(self, channel=3, kernel=3, padding=0, stride=1,
                initializer=None, weight_decay=None, ignore_bias=False):
        self._chnls = channel
        self._krnl = kernel
        self._pdng = padding
        self._strd = stride
        self._init = initializer

        self.params['w'] = graph_variable(weight_decay=weight_decay)
        self.params['b'] = graph_variable(allow_update=not ignore_bias)

    def connect(self, other):
        ret = DeconvolutionalGraph(self._chnls, self._krnl, self._pdng, self._strd, self._init, previous_element=[
                                   other, self.params['w'], self.params['b']])
        return ret
