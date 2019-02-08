import renom as rm
from renom.layers.function.utils import im2col, col2im, imncol, colnim, colnw
from renom.graph.core import operation, UserGraph, GraphMultiStorage, GraphFactory, graph_variable
import renom.utility.initializer as init
import numpy as np


class Conv(GraphFactory):

    """Convolutional Layer.

      This class creates a convolution filter to be convolved with
      the input tensor. This class accepts up to 3d image input.
      Note that the 2d implementation differs slightly from the 3d implementation, giving no
      guarantee that they will perform equally.

      Args:
          channel (int): The dimensionality of the output.
          filter (int): Filter size of the convolution kernel.
          padding (int): Size of the zero-padding around the image.
          stride (int): Stride-size of the convolution.
          initializer (Initializer): Initializer object for weight initialization.

      Example:
          In [1]: import numpy as np
          In [2]: import renom as rm
          In [3]: n, c, h, w = (10, 3, 32, 32)
          In [4]: x = np.random.rand(n, c, h, w)
          In [5]: x.shape
          Out[5]: (10, 3, 32, 32)
          In [6]: layer = rm.graph.Conv(channels = 5)
          In [7]: z = layer(x).as_ndarray()
          In [8]: z.shape
          Out[8]: (10, 5, 30, 30)

      Note:
          Tensor data format is **NCHW**.
    """

    def __init__(self, channels=3, kernel=3, padding=0, stride=1,
                 initializer=None, weight_decay=None, ignore_bias=False):
        super().__init__()
        self._chnls = channels
        self._krnl = kernel
        self._pdng = padding
        self._strd = stride
        self._init = initializer
        self.params['w'] = graph_variable(weight_decay=weight_decay)
        self.params['b'] = graph_variable(allow_update=not ignore_bias)

    def connect(self, other):
        ret = ConvElement(self._chnls, self._krnl, self._pdng, self._strd,
                          self._init, previous_element=[other, self.params['w'], self.params['b']])
        return ret


class conv_forward(operation):

    name = 'Convolution (F)'
    consumes = ['w', 'b']
    workspace_size = 0
    workspace = None

    def __init__(self, channels, kernel=3, padding=0, stride=1, initializer=None):
        self._channels = channels
        self._k = kernel
        self._p = padding
        self._s = stride
        self._d = 1
        self._init = initializer

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
        if self._init is None:
            self._init = init.GlorotNormal() if dims == 2 else init.Gaussian()
        gpus = inputs.gpus
        self.gpus = gpus

        weight_shape = (self._channels, input_shape[1], *self._kernel)
        bias_shape = (1, self._channels, *(1 for i in range(dims)))

        weights.__init__(shape=weight_shape, gpus=gpus, initializer=self._init)
        bias.__init__(shape=bias_shape, gpus=gpus, initializer=init.Constant(0))

        self._weights = weights
        self._bias = bias

        imgs = tuple((input_shape[i + 2] + self._padding[i] * 2 -
                      self._kernel[i]) // self._stride[i] + 1 for i in range(dims))
        output_shape = [input_shape[0], self._channels, *imgs]
        self._outputs = GraphMultiStorage(shape=output_shape, gpus=gpus)
        self._vars = {'w': self._weights, 'b': self._bias, 'y': self._outputs}

        if rm.is_cuda_active():
            with rm.cuda.RenomHandler():
                if dims == 2:
                    self._conv_desc = rm.cuda.ConvolutionDescriptor(
                        self._padding, self._stride, self._dilation, rm.precision)
                    self._filter_desc = rm.cuda.FilterDescriptor(weight_shape, rm.precision)
                else:
                    self._conv_desc = rm.cuda.ConvolutionNDescriptor(
                        self._padding, self._stride, rm.precision)
                    self._filter_desc = rm.cuda.NdFilterDescriptor(weight_shape, rm.precision)
                self._info = [0]
                self._bwd_info = {'data': [0], 'filter': [0]}

    def optimize(self):
        if not rm.is_cuda_active():
            return True
        with rm.cuda.RenomHandler() as handle:
            self._info = rm.cuda.cuGetConvolutionFwdInfo(
                handle, self._conv_desc, self._filter_desc, self._inputs[0], self._outputs[0])
            self._bwd_info = rm.cuda.cuGetConvolutionBwdInfo(
                handle, self._conv_desc, self._filter_desc, self._inputs[0], self._outputs[0])
            req_sz = max(self._info[1], self._bwd_info['data'][1], self._bwd_info['filter'][1])

        if req_sz > conv_forward.workspace_size:
            conv_forward.workspace_size = req_sz
        return True

    def finalize(self):
        if conv_forward.workspace is None:
            workspace_shape = (
                int(np.ceil(conv_forward.workspace_size / np.dtype(rm.precision).itemsize)),)
            conv_forward.workspace = GraphMultiStorage(shape=workspace_shape, gpus=self.gpus)

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            workspace = (conv_forward.workspace_size,
                         conv_forward.workspace[gpu]) if conv_forward.workspace_size > 0 else None
            rm.cuda.cuConvolutionForwardBiasActivation(
                handle, self._conv_desc, self._filter_desc, self._inputs[gpu],
                self._weights[gpu], self._outputs[gpu], self._bias[gpu], self._info[0], workspace)


class conv_forward_cpu(conv_forward):

    def perform(self):
        x = self._inputs['cpu']
        w = self._weights['cpu']
        b = self._bias['cpu']
        if self._dims == 2:
            col = im2col(x, self._outputs.shape[2:], self._kernel,
                         self._stride, self._padding, self._dilation)
            self._col = col
            val = np.rollaxis(np.tensordot(col, w, ([1, 2, 3], [1, 2, 3])), 3, 1)
            ret = val + b
        else:
            col = imncol(x, w, self._stride, self._padding)
            ret = col + b
        self._outputs['cpu'] = ret


class conv_backward(operation):

    name = 'Convolution (B)'
    produces = ['w', 'b']

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):

        inputs = inputs[0]['y']
        self._inputs = inputs
        self._dims = self._fwd_op._dims
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
            self._algo = {'data': 0, 'filter': 0}

    def finalize(self):
        if not rm.is_cuda_active():
            return
        self._algo = {'data': self._fwd_op._bwd_info['data']
                      [0], 'filter': self._fwd_op._bwd_info['filter'][0]}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            workspace = (conv_forward.workspace_size,
                         conv_forward.workspace[gpu]) if conv_forward.workspace_size > 0 else None
            rm.cuda.cuActivationBackward(handle, self._fwd_op._outputs[gpu], self._inputs[gpu])
            rm.cuda.cuConvolutionBackward(handle, self._fwd_op._conv_desc, self._fwd_op._filter_desc,
                                          self._fwd_in[gpu], self._fwd_w[gpu], self._inputs[gpu],
                                          self._weights_out[gpu], self._bias_out[gpu], self._outputs[gpu],
                                          self._algo, workspace)


class conv_backward_cpu(conv_backward):

    def perform(self):
        x = self._fwd_in['cpu']
        w = self._fwd_w['cpu']
        b = self._fwd_b['cpu']
        dy = self._inputs['cpu']
        if self._dims == 2:
            dx = np.tensordot(w, dy, (0, 1))
            dx = np.rollaxis(dx, 3)
            dx = col2im(dx, self._fwd_in.shape[2:], self._fwd_op._stride,
                        self._fwd_op._padding, self._fwd_op._dilation)
            dw = np.tensordot(dy, self._fwd_op._col, ([0, 2, 3], [0, 4, 5]))
            db = np.sum(dy, (0, 2, 3), keepdims=True)
        else:
            dx = colnim(dy, w, self._fwd_op._stride)
            dw = colnw(x, dy, self._fwd_op._stride)
            db = np.sum(dy, axis=tuple(
                [0, ] + [i for i in range(2, len(b.shape))]), keepdims=True)
        self._outputs['cpu'] = dx
        self._weights_out['cpu'] = dw
        self._bias_out['cpu'] = db


class ConvElement(UserGraph):

    def __init__(self, channels=3, kernel=3, padding=0, stride=1, initializer=None, previous_element=None):

        self._chnls = channels
        self._krnl = kernel
        self._pdng = padding
        self._strd = stride
        fwd_op = conv_forward(channels, kernel, padding, stride, initializer) if rm.is_cuda_active(
        ) else conv_forward_cpu(channels, kernel, padding, stride, initializer)
        bwd_ops = [conv_backward(fwd_op) if rm.is_cuda_active() else conv_backward_cpu(fwd_op)]

        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_element)


def del_workspace():
    conv_forward.workspace = None


import atexit
atexit.register(del_workspace)
