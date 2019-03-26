import renom as rm
from renom.utils import im2col, col2im, imnpool, poolnim
from renom.graph.core import operation, UserGraph, GraphMultiStorage, GraphFactory
import numpy as np


class pool_forward(operation):

    name = 'Pool (F)'

    def __init__(self, kernel=3, padding=0, stride=1, mode='max'):
        self._k = kernel
        self._p = padding
        self._s = stride
        self._mode = mode

    def setup(self, inputs):

        inputs = inputs[0]['y']
        input_shape = inputs.shape
        dims = len(input_shape[2:])
        self._dims = dims

        self._inputs = inputs
        self._kernel = np.array(list(self._k for i in range(dims))).astype(np.int32)
        self._padding = np.array(list(self._p for i in range(dims))).astype(np.int32)
        self._stride = np.array(list(self._s for i in range(dims))).astype(np.int32)

        imgs = tuple((input_shape[i + 2] + self._padding[i] * 2 -
                      self._kernel[i]) // self._stride[i] + 1 for i in range(dims))
        out_shape = [input_shape[0], input_shape[1], *imgs]
        self.gpus = inputs.gpus
        outs = GraphMultiStorage(shape=out_shape, gpus=self.gpus)
        self._outputs = outs
        self._vars = {'y': outs}
        if rm.is_cuda_active():
            if dims == 2:
                pd = rm.cuda.PoolingDescriptor(
                    self._kernel, self._padding, self._stride, pool_mode=0 if self._mode == 'max' else 1)
            else:
                pd = rm.cuda.PoolingNDescriptor(
                    self._kernel, self._padding, self._stride, pool_mode=0 if self._mode == 'max' else 1)
            self._pool_desc = pd

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cuPoolingForward(handle, self._pool_desc, self._inputs[gpu], self._outputs[gpu])


class pool_forward_cpu(pool_forward):

    def perform(self):
        x = self._inputs['cpu']
        if self._dims == 2 and False:
            col = im2col(x, self._outputs.shape[2:], self._kernel, self._stride, self._padding)
            n, ic, kh, kw, oh, ow = col.shape
            col = col.reshape(n, ic, kh * kw, oh, ow)
            index = np.argmax(col, axis=2)
            self._index = index
            ret = np.max(col, axis=2)
        else:
            ret = imnpool(x, self._kernel, self._stride, self._padding, mode=self._mode)
        self._outputs['cpu'] = ret


class pool_backward(operation):

    name = 'Pool (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):

        inputs = inputs[0]['y']
        self._inputs = inputs
        out_shape = self._fwd_op._inputs.shape
        self._fwd_in = self._fwd_op._inputs
        self._fwd_out = self._fwd_op._outputs
        self.gpus = inputs.gpus
        outs = GraphMultiStorage(shape=out_shape, gpus=self.gpus)
        self._outputs = outs
        self._vars = {'y': outs, id(self._fwd_in): outs}

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cuPoolingBackward(handle, self._fwd_op._pool_desc,
                                      self._fwd_in[gpu], self._fwd_out[gpu], self._inputs[gpu], self._outputs[gpu])


class pool_backward_cpu(pool_backward):

    def perform(self):
        dims = self._fwd_op._dims
        dy = self._inputs['cpu']
        N = len(dy)
        x = self._fwd_op._inputs['cpu']
        if dims == 2 and False:
            index = self._fwd_op._index
            in_shape = self._fwd_op._inputs.shape
            out_shape = self._fwd_op._outputs.shape
            col = np.zeros((N, in_shape[1], self._fwd_op._kernel[0], self._fwd_op._kernel[1],
                            out_shape[2], out_shape[3]))
            col_k = np.rollaxis(col.reshape(N, in_shape[1], -1,
                                            out_shape[2], out_shape[3]), 2)
            for i in np.ndindex(N, in_shape[1], out_shape[2], out_shape[3]):
                col_k[index[i]][i] = dy[i]
            dx = col2im(col, in_shape[2:], self._fwd_op._stride, self._fwd_op._padding)
        else:
            dx = poolnim(x, dy, self._fwd_op._kernel, self._fwd_op._stride,
                         self._fwd_op._padding, mode=self._fwd_op._mode)
        self._outputs['cpu'] = dx


class PoolElement(UserGraph):

    def __init__(self, kernel, padding, stride, mode, previous_element=None):
        self._krnl = kernel
        self._pad = padding
        self._strd = stride
        fwd_op = pool_forward(kernel, padding, stride, mode) if rm.is_cuda_active() \
            else pool_forward_cpu(kernel, padding, stride, mode)

        bwd_ops = [pool_backward(fwd_op) if rm.is_cuda_active() else pool_backward_cpu(fwd_op)]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_element)


class PoolGraphFactory(GraphFactory):
    '''General initializer for pooling functions.

    Args:
        kernel (int, tuple): The size of the kernel to apply on the images \
        accross all dimensions
        padding (int, tuple): The size of the padding to add to the edges of \
        each image.
        stride (int, tuple): The step size between the points where the \
        kernels are to be applied.

    '''

    def prepare(self, kernel=3, padding=0, stride=1):
        self._krnl = kernel
        self._pad = padding
        self._strd = stride


class MaxPool(PoolGraphFactory):
    '''Max pooling function.

    This function takes max operation for each cells overlapped by the filter kernel.

    This function accepts input array which has 2~5 dimention.
    If argments of kernel, padding or strideis are given as int, it will be
    expanded to fit the dimension of the input array.

    Args:
        kernel (int, tuple): The size of the kernel to apply on the images \
        accross all dimensions
        padding (int, tuple): The size of the padding to add to the edges of \
        each image.
        stride (int, tuple): The step size between the points where the \
        kernels are to be applied.

    '''

    def connect(self, other):
        ret = PoolElement(self._krnl, self._pad, self._strd, mode='max', previous_element=[other])
        return ret


class AvgPool(PoolGraphFactory):
    '''Average pooling function.
    This function takes average for each cells overlapped by the filter kernel.

    This function accepts input array which has 2~5 dimention.
    If argments of kernel, padding or strideis are given as int, it will be
    expanded to fit the dimension of the input array.

    Args:
        kernel (int, tuple): The size of the kernel to apply on the images \
        accross all dimensions
        padding (int, tuple): The size of the padding to add to the edges of \
        each image.
        stride (int, tuple): The step size between the points where the \
        kernels are to be applied.

    '''

    def connect(self, other):
        ret = PoolElement(self._krnl, self._pad, self._strd,
                          mode='average', previous_element=[other])
        return ret
