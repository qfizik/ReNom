import renom as rm
import numpy as np
from .operation import operation
from .operational_element import operational_element
from .graph_storage import GraphMultiStorage
import renom.utility.initializer as init
import types

T = True
F = False


class gradient_accumulator(operation):

    name = 'Gradient Accumulator'

    def setup(self, inputs):
        self._num_grads = len(inputs)
        self._inputs = [grad['dy'] for grad in inputs]
        gpus = self._inputs[0].gpus
        shape = self._inputs[0].shape
        self.gpus = gpus
        if self._num_grads == 1:
            out = self._inputs[0]
        else:
            assert all(grad.shape == shape for grad in self._inputs)
            out = GraphMultiStorage(shape=shape, gpus=self.gpus)
        self._outputs = out
        self._vars = {'y': out, 'dy': out}

    def perform(self):
        if self._num_grads == 1:
            pass
        else:
            if rm.is_cuda_active():
                self.accumulate_gpu()
            else:
                self.accumulate_cpu()

    def accumulate_gpu(self):
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            rm.cu.cusub(self._outputs[gpu], self._outputs[gpu],
                        self._outputs[gpu], handle)
            for grad in self._inputs:
                rm.cu.cuadd(self._outputs[gpu], grad[gpu],
                            self._outputs[gpu], handle)

    def accumulate_cpu(self):
        dy = 0
        for grad in self._inputs:
            dy += grad['cpu']
        self._outputs['cpu'] = dy


class update_operation(operation):
    '''
        The update operation is a special type of operation, designed in such a way that its contents can
        be replaced at any time. A normal operation functions through connecting to its inputs and outputs,
        as determined by the graph of operational elements that contain the operation.

        In order to connect the update operation, the UserGraph feeds it three components, the consumer,
        the producer and a shared key. The update operation then assumes that the gradient found under
        the key locatedin the producer should be applied to the value in the consumer with the same key.

        To change the mode of operation, the update_operation receives an optimizer_factory type object,
        which produces the setup and perform functionality of the update operation.
        Graphs would otherwise have to rebuild with a different operation, through this optimizer_factory
        delivering the operation functionality, we can keep the same update_operation in the graph but
        allow it to change during training.
    '''
    roles = ['update']
    name = 'Update'
    _communicator = None
    _should_update = True

    def __init__(self, consumer, producer, key, operation=None):
        # if operation is None:
        #  operation = Sgd(0.01, 0.4) if rm.is_cuda_active() else sgd_update_cpu(0.01, 0.4)
        self._consumer = consumer
        self._producer = producer
        self._shared_key = key
        self._update_op = operation
        self._factory = None
        self.name += " ({} of {})".format(key, consumer.name[:-4])

    def set_update_op(self, fac):
        if self._factory is fac:
            return
        self._factory = fac

    def setup(self, inputs):
        if self._factory is None:
            self._factory = rm.graph.Sgd(1.0, 0.0)
        assert self._factory is not None
        #self._dy = self._producer.get_key(self._shared_key)
        if self._shared_key in inputs[0]:
            self._dy = inputs[0][self._shared_key]
        else:
            self._dy = inputs[0]['y']
        self._outputs = self._consumer.get_key(self._shared_key)
        self._wd = None  # For weight decay
        self._vars = {'y': self._dy}
        gpus = self._outputs.gpus
        self.gpus = gpus
        if self._update_op is None:
            self._update_op = self._factory.get_op(self._dy, self._outputs)
            self._update_op.setup(self._dy, self._outputs)
        if update_operation._communicator is None and not isinstance(self.gpus, str) and len(self.gpus) > 1:
            update_operation._communicator = rm.cuda.DeviceCommunicator(len(self.gpus))

    def check_weight_decay(self):
        if self._outputs._weight_decay is not None:
            wd = self._outputs._weight_decay
            if rm.cuda.is_cuda_active():
                if self._wd is None:
                    self._wd = GraphMultiStorage(shape=self._dy.shape, gpus=self.gpus)
                for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
                    rm.cuda.cumul(self._outputs[gpu], wd, self._wd[gpu], handle)
                    rm.cuda.cuadd(self._dy[gpu], self._wd[gpu], self._dy[gpu], handle)
            else:
                self._dy['cpu'] = self._dy['cpu'] + self._outputs['cpu'] * wd

    def perform(self):
        if update_operation._communicator is not None:
            update_operation._communicator.allReduce(self._dy)
        if self._outputs._should_update and self._should_update:
            self.check_weight_decay()
            self._update_op.update()
