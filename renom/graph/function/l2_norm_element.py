from renom.graph.core import UserGraph, operational_element, operation, GraphMultiStorage, GraphFactory, graph_variable
import renom.utility.initializer as init
import renom as rm
import numpy as np


class l2norm_forward(operation):

    name = 'L2Norm (F)'
    consumes = ['w']

    def __init__(self, scale):

        self._scale = scale

    def setup(self, inputs):
        weights = inputs[1]['y']
        inputs = inputs[0]['y']
        assert isinstance(inputs, GraphMultiStorage), 'Received {}'.format(type(inputs))
        self.gpus = inputs.gpus
        self._inputs = inputs
        weight_shape = (1, inputs.shape[1], 1, 1)
        weights.__init__(shape=weight_shape, gpus=self.gpus, initializer=init.Constant(self._scale))
        output_shape = inputs.shape
        outputs = GraphMultiStorage(shape=output_shape, gpus=self.gpus)
        self._vars = {'x': inputs, 'w': weights, 'y': outputs}
        self._weights = weights
        self._outputs = outputs

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            norm = rm.cuda.cusum(
                (self._inputs[gpu] * self._inputs[gpu]), handle, axis=1, keepdims=True)
            rm.cuda.cusqrt(norm, norm)
            rm.cuda.cuadd(norm, 1e-7, norm, handle)
            z = (self._inputs[gpu] / norm) * self._weights[gpu]
            self._outputs[gpu].copy_from(z)


class l2norm_forward_cpu(l2norm_forward):

    def perform(self):
        x = self._inputs['cpu']
        w = self._weights['cpu']
        scale = self._scale

        norm = np.sqrt(np.sum(x * x, axis=1, keepdims=True)) + 1e-7
        self._norm = norm
        ret = (x / norm) * w
        self._outputs['cpu'] = ret


class l2norm_backward(operation):

    name = 'L2Norm (B)'

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):

        inputs = inputs[0]['dy']
        gpus = inputs.gpus
        self.gpus = gpus
        weights = self._fwd_op.get_key('w')
        self._inputs = inputs
        self._weights = weights

        fwd_ins = self._fwd_op.get_key('x')
        output_shape = fwd_ins.shape

        outputs = GraphMultiStorage(shape=output_shape, gpus=gpus, initializer=None)
        tmp = GraphMultiStorage(shape=output_shape, gpus=gpus)

        self._vars = {'y': outputs, 'dy': outputs, id(fwd_ins): outputs}
        self._outputs = outputs
        self._fwd_ins = fwd_ins
        self._tmp = tmp

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            norm = rm.cuda.cusum(
                (self._fwd_ins[gpu] * self._fwd_ins[gpu]), handle, axis=1, keepdims=True)
            rm.cuda.cusqrt(norm, norm)
            norm = norm + 1e-7
            dx = self._inputs[gpu] * norm - \
                rm.cuda.cusum(self._fwd_ins[gpu] * self._inputs[gpu], handle,
                              axis=1, keepdims=True) * self._fwd_ins[gpu] / norm
            dx = dx / (norm * norm)
            dx = dx * self._weights[gpu]
            self._outputs[gpu].copy_from(dx)


class l2norm_backward_cpu(l2norm_backward):

    def perform(self):
        dy = self._inputs['cpu']
        w = self._weights['cpu']
        x = self._fwd_ins['cpu']
        norm = self._fwd_op._norm
        dx = dy * norm - (np.sum(x * dy, axis=1, keepdims=True) * x) / norm
        dx = dx / (norm * norm)
        ret = dx * w
        self._outputs['cpu'] = ret


class l2norm_weight_backward(operation):

    name = 'L2Norm Weight (B)'
    produces = ['w']

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        inputs = inputs[0]['dy']
        self._inputs = inputs

        gpus = inputs.gpus
        self.gpus = gpus
        fwd_ins = self._fwd_op.get_key('x')
        fwd_weights = self._fwd_op.get_key('w')
        output_shape = fwd_weights.shape

        outputs = GraphMultiStorage(shape=output_shape, gpus=gpus, initializer=None)

        self._vars = {'y': outputs, 'w': outputs, id(fwd_weights): outputs}

        self._fwd_ins = fwd_ins
        self._outputs = outputs

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            norm = rm.cuda.cusum(
                (self._fwd_ins[gpu] * self._fwd_ins[gpu]), handle, axis=1, keepdims=True)
            rm.cuda.cusqrt(norm, norm)
            norm = norm + 1e-7
            dl = self._inputs[gpu] * (self._fwd_ins[gpu] / norm)
            tmp = rm.cuda.cusum(dl, handle, axis=(0, 2, 3), keepdims=True)
            self._outputs[gpu].copy_from(tmp)


class l2norm_weight_backward_cpu(l2norm_weight_backward):

    def perform(self):
        x = self._fwd_ins['cpu']
        dy = self._inputs['cpu']
        norm = self._fwd_op._norm
        dl = dy * (x / norm)
        ret = np.sum(dl, axis=(0, 2, 3), keepdims=True)
        self._outputs['cpu'] = ret


class L2NormGraph(UserGraph):

    has_back = True

    def __init__(self, scale, previous_element=None):

        fwd_op = l2norm_forward(scale) if rm.is_cuda_active() else l2norm_forward_cpu(scale)
        bwd_ops = [l2norm_backward(associated_forward=fwd_op) if rm.is_cuda_active() else l2norm_backward_cpu(fwd_op),
                   l2norm_weight_backward(associated_forward=fwd_op) if rm.is_cuda_active(
        ) else l2norm_weight_backward_cpu(fwd_op)
        ]
        self.scale = scale

        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_element)


class L2NormGraphElement(GraphFactory):

    def __init__(self, scale=20):
        super().__init__()
        self.scale = scale
        self.params['w'] = graph_variable()

    def connect(self, other):
        ret = L2NormGraph(scale=self.scale, previous_element=[other, self.params['w']])
        return ret
