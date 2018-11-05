import renom as rm
from .operation import operation
from .graph_element import operational_element

class update_operation(operation):

  name = 'Update Operation'
  _communicator = None

  def __init__(self, learning_rate, consumer, producer, key):
    self._lr = learning_rate
    self._setup = False
    self._consumer = consumer
    self._producer = producer
    self._shared_key = key

  def setup(self, inputs, storage):
    self._dy = self._producer.get_key(self._shared_key)
    self._outputs = self._consumer.get_key(self._shared_key)
    gpus = self._outputs.gpus
    self.gpus = gpus
    if update_operation._communicator is None and len(self.gpus) > 1:
      update_operation._communicator = rm.cuda.DeviceCommunicator(len(gpus))
    self._setup = True    

  def perform(self):
    if len(self.gpus) > 1:
      update_operation._communicator.allReduce(self._dy)
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cu_optimizer_sgd(self._lr, 0, self._dy[gpu], None, self._outputs[gpu], handle)

  def get_output_signature(self): return self._outputs

    
