import renom as rm
from renom.graph.core import learnable_graph_element, operation, GraphFactory, graph_variable, multi_gpu_variable


class _forward_operation(operation):

  def setup(self, inputs, storage):
    pass

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      pass

class _backward_operation(operation):

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):
    pass

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      pass


class _graph_element(learnable_graph_element):

  def __init__(self, previous_elements = None):
    fwd_op = _forward_operation()
    bwd_ops = [ _backward_operation(fwd_op) ]
    super().__init__(forward_operation = fwd_op, backward_operations = bwd_ops, previous_elements = previous_elements)


