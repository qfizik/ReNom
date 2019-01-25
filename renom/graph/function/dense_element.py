from renom.graph.core import UserGraph, operational_element, operation, GraphMultiStorage, GraphFactory, graph_variable
import renom.utility.initializer as init
import renom as rm
import numpy as np


class DenseGraphElement(GraphFactory):
    '''Fully connected layer as described below.

        See also :py:class:`~renom.layers.function.dense.Dense`
        See also :class:`~renom.graph.function.BiasGraphElement`

          :math:`f(x)= w \cdot x + b`

      Args:
          output_size (int): Output unit size.
          initializer (Initializer): Initializer object for weight initialization.

      Example:

      .. code-block:: python

          In [1]: import numpy as np
          In [2]: import renom as rm
          In [3]: x = np.random.rand(3, 2)
          In [4]: x.shape
          Out[4]: (3, 2)
          In [5]: layer = rm.graph.DenseGraphElement(3)
          In [6]: z = layer(x).as_ndarray()
          In [7]: z.shape
          Out[7]: (3, 3)
    '''

    def __init__(self, output_size=3, initializer=None, weight_decay=None, ignore_bias=False):
        super().__init__()
        self.output_size = output_size
        self.params['w'] = graph_variable(weight_decay=weight_decay)
        self._ignore_bias = ignore_bias
        if not ignore_bias:
            self._bias = rm.graph.BiasGraphElement()
            self.params['b'] = self._bias.params['b']
        self._init = initializer

    def connect(self, other):
        ret = DenseGraph(output_size=self.output_size, initializer=self._init,
                         previous_element=[other, self.params['w']])
        if not self._ignore_bias:
            ret = self._bias(ret)
        return ret


class DenseGraph(UserGraph):

    def __init__(self, output_size, initializer, previous_element=None):

        fwd_op = dense_forward(output_size, initializer) if rm.is_cuda_active(
        ) else dense_forward_cpu(output_size, initializer)
        bwd_ops = [dense_backward(associated_forward=fwd_op) if rm.is_cuda_active() else dense_backward_cpu(fwd_op),
                   dense_weight_backward(associated_forward=fwd_op) if rm.is_cuda_active(
        ) else dense_weight_backward_cpu(fwd_op),
        ]
        self.output_size = output_size

        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_element)


class dense_forward(operation):

    name = 'Dense (F)'
    consumes = ['w']

    def __init__(self, output_size, initializer):
        self._output_size = output_size
        self._init = initializer

    def setup(self, inputs):
        weights = inputs[1]['y']
        inputs = inputs[0]['y']
        assert isinstance(inputs, GraphMultiStorage), 'Received {}'.format(type(inputs))
        self.gpus = inputs.gpus
        self._inputs = inputs
        if self._init is None:
            self._init = init.GlorotNormal()
        weight_shape = (inputs.shape[1], self._output_size)
        weights.__init__(shape=weight_shape, gpus=self.gpus, initializer=self._init)
        output_shape = (inputs.shape[0], self._output_size)
        outputs = GraphMultiStorage(shape=output_shape, gpus=self.gpus)
        self._vars = {'x': inputs, 'w': weights, 'y': outputs}
        self._weights = weights
        self._outputs = outputs

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cublas_gemm(self._inputs[gpu], 0,
                                self._weights[gpu], 0, self._outputs[gpu], handle)


class dense_forward_cpu(dense_forward):

    def perform(self):
        ret = np.dot(self._inputs['cpu'], self._weights['cpu'])
        self._outputs['cpu'] = ret


class dense_backward(operation):

    name = 'Dense (B)'

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

        self._vars = {'y': outputs, 'dy': outputs, id(fwd_ins): outputs}
        self._outputs = outputs

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cuda.cublas_gemm(self._inputs[gpu], 0,
                                self._weights[gpu], 1, self._outputs[gpu], handle)


class dense_backward_cpu(dense_backward):

    def perform(self):
        ret = np.dot(self._inputs['cpu'], self._weights['cpu'].T)
        self._outputs['cpu'] = ret


class dense_weight_backward(operation):

    name = 'Dense Weight (B)'
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
            rm.cuda.cublas_gemm(self._fwd_ins[gpu], 1,
                                self._inputs[gpu], 0, self._outputs[gpu], handle)


class dense_weight_backward_cpu(dense_weight_backward):

    def perform(self):
        ret = np.dot(self._fwd_ins['cpu'].T, self._inputs['cpu'])
        self._outputs['cpu'] = ret
