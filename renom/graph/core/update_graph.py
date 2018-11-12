import renom as rm
from .operation import operation
from .graph_element import operational_element
from .new_gpu import multi_gpu_variable
import renom.utility.initializer as init

class update_operation(operation):

  name = 'Update Operation'
  _communicator = None

  def __init__(self, learning_rate, consumer, producer, key):
    self._lr = learning_rate
    self._consumer = consumer
    self._producer = producer
    self._shared_key = key

  def setMethod(self, method):
  
    if method == 'sgd':
      self._extras = { }
      self._extras['running_average'] = multi_gpu_variable(shape = self._dy.shape, gpus = self._dy.gpus, initializer = init.Constant(0))

      def perform():
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
          rm.cuda.cu_optimizer_sgd(self._lr, 0, self._dy[gpu], self._extras['running_average'][gpu], self._outputs[gpu], handle)

      self.perform = perform
    else:
      raise Exception('Unknown method {}'.format(method))


  def setup(self, inputs, storage):
    self._storage = storage
    updates = storage.retrieve('Updates')
    if updates is None:
      updates = [ self ]
      storage.register('Updates', updates)
    else:
      if self not in updates:
        updates.append(self)
    self._dy = self._producer.get_key(self._shared_key)
    self._outputs = self._consumer.get_key(self._shared_key)
    gpus = self._outputs.gpus
    self.gpus = gpus
    method = storage.retrieve('Update_Method')
    if method is None:
      method = 'sgd'
    self.setMethod(method)
    if update_operation._communicator is None and len(self.gpus) > 1:
      update_operation._communicator = rm.cuda.DeviceCommunicator(len(gpus))

  def perform(self): raise NotImplementedError

  def get_output_signature(self): return self._outputs

    
