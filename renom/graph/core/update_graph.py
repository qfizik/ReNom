import renom as rm
from .operation import operation
from .operational_element import operational_element
from .graph_storage import GraphMultiStorage
import renom.utility.initializer as init
import types

T = True
F = False


class update_operation(operation):

    name = 'Update Operation'
    roles = ['update']
    _communicator = None

    def __init__(self, consumer, producer, key, operation=None):
        # if operation is None:
        #  operation = sgd_update(0.01, 0.4) if rm.is_cuda_active() else sgd_update_cpu(0.01, 0.4)
        self._consumer = consumer
        self._producer = producer
        self._shared_key = key
        self._update_op = operation
        self._factory = None

    def set_update_op(self, fac):
        if self._factory is fac:
            return
        self._factory = fac

    def setup(self, inputs):
        if self._factory is None:
            self._factory = rm.graph.sgd_update()
        assert self._factory is not None
        self._dy = self._producer.get_key(self._shared_key)
        self._outputs = self._consumer.get_key(self._shared_key)
        self._vars = {'y': self._dy}
        gpus = self._outputs.gpus
        self.gpus = gpus
        if self._update_op is None:
            self._update_op = self._factory.get_op(self._outputs)
            self._update_op.setup(self._dy, self._outputs)

        if update_operation._communicator is None and not isinstance(self.gpus, str) and len(self.gpus) > 1:
            update_operation._communicator = rm.cuda.DeviceCommunicator(len(gpus))

    def perform(self):
        if len(self.gpus) > 1 and F:
            update_operation._communicator.allReduce(self._dy)

        self._update_op.update()
