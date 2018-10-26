import renom as rm
from operation import operation
from graph_element import operational_element

class update_operation(operation):

  name = 'Update Operation'

  def __init__(self, learning_rate, doobie):
    self._lr = learning_rate
    self._setup = False
    self._doob = doobie

  def setup(self, inputs):
    self._outputs = self._doob.get_weights_signature()
    self._dy = inputs[0]
    self._setup = True    

  def perform(self):
    assert self._setup
    with rm.cuda.RenomHandler() as handle:
      rm.cuda.cu_optimizer_sgd(self._lr, 0, self._dy[0], None, self._outputs[0], handle)

  def get_output_signature(self): return self._outputs

    
