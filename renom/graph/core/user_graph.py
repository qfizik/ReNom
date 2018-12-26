from .graph_element import graph_element, operational_element
from .update_graph import update_operation
from .operation import operation
import renom as rm
import numpy as np

def _prepare_prevs(previous_elements):
  if not isinstance(previous_elements, list):
    previous_elements = [ previous_elements ]
  for i, prev in enumerate(previous_elements):
    assert isinstance(prev, np.ndarray) or isinstance(prev, UserGraph)
    if isinstance(prev, np.ndarray):
      previous_elements[i] = rm.graph.StaticVariable(prev)
  return previous_elements



class UserGraph(graph_element):
  '''
    A learnable graph element is responsible for storing and performing the forward, backward and update operations in a normal neural-network setting.
  '''

  _has_back = False
  _name = 'Undefined'

  def __init__(self, forward_operation, backward_operations = None, previous_elements = None):
    self.connected = False
    if backward_operations is None:
      backward_operations = [ ]

    if previous_elements is not None:
      previous_elements = _prepare_prevs(previous_elements)

    super().__init__(previous_elements = previous_elements)

    self._create_fwd_graph(forward_operation)
    self._create_bwd_graphs(backward_operations)
    self._create_update_graphs(forward_operation, backward_operations)

    if previous_elements is not None:
      self.connect(previous_elements = previous_elements)


  # Some helper functions to divide the __init__ method into smaller parts
  def _create_bwd_graphs(self, backward_operations):
    self._bwd_graphs = []
    for op in backward_operations:
      bwd_graph = operational_element(op, tags = ['Backward'])
      self._bwd_graphs.append(bwd_graph)

  def _create_fwd_graph(self, forward_operation):
    assert isinstance(forward_operation, operation) or isinstance(forward_operation, operational_element)
    if isinstance(forward_operation, operation):
      self._fwd = operational_element(operation = forward_operation, tags = ['Forward'])
    elif isinstance(forward_operation, operational_element):
      forward_graph = forward_operation
    else:
      raise AttributeError('Uknown forward operation type')


  def _create_update_graphs(self, forward_operation, backward_operations):
    if isinstance(forward_operation, operation):
      assert len(backward_operations) == len(self._bwd_graphs)
      for consumed in forward_operation.consumes:
        for op_num, op in enumerate(backward_operations):
          if consumed in op.produces:
            upd = update_operation(consumer = forward_operation, producer = op, key = consumed)
            upd_g = operational_element(upd, tags = ['Update'])
            upd_g.add_input(self._bwd_graphs[op_num])


  def connect(self, previous_elements):
    if self.connected is True:
      self.disconnect()
      assert len(self._previous_elements) == 0 and len(self._fwd._previous_elements) == 0

    if isinstance(previous_elements, UserGraph):
      previous_elements = [ previous_elements ]

    for elem in previous_elements:
      prev_graph_input = elem.get_forward_output()
      self._fwd.add_input(prev_graph_input)

    for num, elem in enumerate(previous_elements):
      if elem.has_back:
        elem.connect_back(self, pos = num)
    self.connected = True
    self.simple_forward()
    return self

  def disconnect(self):
    while len(self._previous_elements) > 0:
      self.remove_input(self._previous_elements[0])
    while len(self._fwd._previous_elements) > 0:
      self._fwd.remove_input(self._fwd._previous_elements[0])
    for graph in self._bwd_graphs:
      while len(graph._previous_elements) > 0:
        graph.remove_input(graph._previous_elements[0])
    while len(self._next_elements) > 0:
      self._next_elements[0]._fwd.remove_input(self._fwd)
      self._next_elements[0].remove_input(self)

  def connect_back(self, previous_element, pos = 0):
    backward_graph_input = previous_element.get_backward_output(pos)
    for graph in self._bwd_graphs:
      graph.add_input(backward_graph_input)

  def disconnect_back(self, previous_element, pos = 0):
    backward_graph_input = previous_element.get_backward_output(pos)
    for graph in self._bwd_graphs:
      graph.remove_input(backward_graph_input)


  @property
  def has_back(self):
    return self._has_back

  @property
  def name(self):
    return self._name

  def __call__(self, *args, **kwargs): return self.connect(*args, **kwargs)
  def __repr__(self):
    self.forward()
    return self._fwd.__repr__()

  class Executor:

    def __init__(self, call_list, graph_inputs, losses):
      self.call_list = call_list
      self.dispatchers = graph_inputs
      print(self.dispatchers)
      self.loss = losses

    def execute(self, epochs = None, steps = 1):
      if epochs is None:
        return
      while(epochs > 0):
        try:
          loss = 0
          while(True):
            #rm.cuda.cuDeviceSynchronize()
            self.perform_step()
            loss += self.loss[0].as_ndarray()#.get_loss()
        except StopIteration:
          print(loss)
          for disp in self.dispatchers:
            disp.reset()
          epochs -= 1

    def __del__(self):
      for i in range(len(self.dispatchers)): self.dispatchers[i] = None
      for i in range(len(self.loss)): self.loss[i] = None


    def perform_step(self):
      for depth in self.call_list.keys():
        for call in self.call_list[depth]:
          call()
          print('Forwarding')

    def loss(self):
      return self.loss_func

  def getInferenceExecutor(self):
    ins = self._fwd.gather_operations_with_role('input')
    lss = self._fwd.gather_operations_with_role('loss')
    inputs = []
    for d in ins.keys(): inputs.extend(ins[d])
    losses = []
    for d in lss.keys(): losses.extend(lss[d])
    dct = self._fwd.get_call_dict('Forward')
    ret = UserGraph.Executor(dct, inputs, losses)
    return ret

  def getTrainingExecutor(self, optimizer = None):
    ups = self._bwd_graphs[0].gather_operations_with_role('update')
    for i in range(len(ups)):
      ups[i].set_update_op(optimizer)
      ups[i] = None # Avoiding destruction errors
    ins = self._bwd_graphs[0].gather_operations_with_role('input')
    lss = self._bwd_graphs[0].gather_operations_with_role('loss')
    self._fwd.continue_setup()
    dct = self._bwd_graphs[0].get_call_dict()
    ret = UserGraph.Executor(dct, ins, lss)
    return ret

  def simple_forward(self):
    self._fwd.forward()
    return self

  def forward(self):
    self._fwd.calculate_forward()
    return self

  def optimize(self):
    pass

  def backward(self):
    if len(self._bwd_graphs[0]._previous_elements) == 0:
      loss = rm.graph.ConstantLoss(previous_element = self)
      loss._bwd_graphs[0].add_input(self._fwd)
    self._fwd.continue_forward(tag = 'Backward')
    return self

  def get_gradient(self, some_variable):
    assert isinstance(some_variable, rm.graph.core.GraphMultiStorage)
    search_id = id(some_variable)
    for grph in self._bwd_graphs:
      r = grph._op.get_key(search_id)
      if r is not None:
        return r
    for elem in self._previous_elements:
      r = elem.get_gradient(some_variable)
      if r is not None:
        return r
    raise AttributeError('Could not find {}'.format(search_id))

  def update(self, optimizer=None):
    if optimizer is not None:
      ups = self._bwd_graphs[0].gather_operations_with_role('update')
      for d in ups:
          for i in range(len(ups[d])):
              ups[d][i].set_update_op(optimizer)
              ups[d][i] = None # Avoiding destruction errors
    self._fwd.continue_forward(tag = 'update')

  def print_tree(self):
    #print('I am a {0:s} at depth {1:d}'.format(self.name, self.depth))
    self._fwd.print_tree()

  def get_forward_output(self): return self._fwd
  def get_backward_output(self, num = 0): return self._bwd_graphs[num]

  @property
  def output(self): return self._fwd.output

  def as_ndarray(self):
    self.forward()
    return self._fwd.as_ndarray()


class UserLossGraph(UserGraph):

  def connect(self, previous_elements):
    assert isinstance(previous_elements, list) and len(previous_elements) == 2
    super().connect(previous_elements)
    for elem in previous_elements:
      prev = elem.get_forward_output()
      self._bwd_graphs[0].add_input(prev)
    self._bwd_graphs[0].add_input(self._fwd)
    return self

  def disconnect(self): raise NotImplementedError()
