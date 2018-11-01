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
    gpus = self._outputs._num_gpus
    self._num_gpus = gpus
    if update_operation._communicator is None and self._num_gpus > 1:
      update_operation._communicator = rm.cuda.DeviceCommunicator(gpus)
    self._dy = inputs[0]
    self._setup = True    

  def perform(self):
    assert self._setup
    if self._num_gpus > 1:
      update_operation._communicator.allReduce(self._dy)
    for gpu, handle in enumerate(rm.cuda.RenomHandlers(self._num_gpus)):
      rm.cuda.cu_optimizer_sgd(self._lr, 0, self._dy[gpu], None, self._outputs[gpu], handle)

  def get_output_signature(self): return self._outputs

    
