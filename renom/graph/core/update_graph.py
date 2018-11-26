import renom as rm
from .operation import operation
from .graph_element import operational_element
from .new_gpu import multi_gpu_variable
import renom.utility.initializer as init
import types

T = True
F = False

class sgd_update:

  def __init__(self, learning_rate, momentum):
    self.learning_rate = learning_rate
    self.momentum = momentum

  def setup(self, grad, val):

    self.gpus = grad.gpus
    self._dy = grad
    self._outputs = val
    self._run_avg = multi_gpu_variable(shape = grad.shape, gpus = grad.gpus, initializer = init.Constant(0))


  def update(self):  
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cu_optimizer_sgd(self.learning_rate, self.momentum, self._dy[gpu], self._run_avg[gpu], self._outputs[gpu], handle)

class update_operation(operation):

  name = 'Update Operation'
  _communicator = None

  def __init__(self, consumer, producer, key, operation = None):
    if operation is None:
      operation = sgd_update(0.01, 0.4)
    self._consumer = consumer
    self._producer = producer
    self._shared_key = key
    self._update_op = operation


  def setup(self, inputs, storage):
    self._dy = self._producer.get_key(self._shared_key)
    self._outputs = self._consumer.get_key(self._shared_key)
    gpus = self._outputs.gpus
    self.gpus = gpus
    self.updates = 0

    self._update_op.setup(self._dy, self._outputs)

    if update_operation._communicator is None and not isinstance(self.gpus, str) and  len(self.gpus) > 1:
      update_operation._communicator = rm.cuda.DeviceCommunicator(len(gpus))

  def perform(self):
    if len(self.gpus) > 1 and F:
      update_operation._communicator.allReduce(self._dy)

    self._update_operation.update()

    self.updates += 1
    if len(self.gpus) > 1 and self.updates >= 10 and F:
      update_operation._communicator.allReduce(self._outputs)
      for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
        rm.cuda.cudiv(self._outputs[gpu], len(self.gpus), self._outputs[gpu], handle)
      self.updates = 0

  def get_output_signature(self): return self._outputs

    
