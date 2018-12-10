import renom as rm
from .operation import operation
from .graph_element import operational_element
from .new_gpu import multi_gpu_variable
import renom.utility.initializer as init
import types

T = True
F = False


class update_operation(operation):

  name = 'Update Operation'
  roles = [ 'update' ]
  _communicator = None

  def __init__(self, consumer, producer, key, operation = None):
    #if operation is None:
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

  def setup(self, inputs, storage):
    assert self._factory is not None
    self._dy = self._producer.get_key(self._shared_key)
    self._outputs = self._consumer.get_key(self._shared_key)
    gpus = self._outputs.gpus
    self.gpus = gpus
    self.updates = 0
    if self._update_op is None:
      self._update_op = self._factory.get_op(self._outputs)
      self._update_op.setup(self._dy, self._outputs)

    if update_operation._communicator is None and not isinstance(self.gpus, str) and  len(self.gpus) > 1 and F:
      update_operation._communicator = rm.cuda.DeviceCommunicator(len(gpus))

  def perform(self):
    if len(self.gpus) > 1 and F:
      update_operation._communicator.allReduce(self._dy)

    self._update_op.update()

    self.updates += 1
    if len(self.gpus) > 1 and self.updates >= 10 and F:
      update_operation._communicator.allReduce(self._outputs)
      for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
        rm.cuda.cudiv(self._outputs[gpu], len(self.gpus), self._outputs[gpu], handle)
      self.updates = 0


    
