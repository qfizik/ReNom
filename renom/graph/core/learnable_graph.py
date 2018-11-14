from .graph_element import graph_element, operational_element
from .update_graph import update_operation
from .operation import operation
import renom as rm
import numpy as np

class learnable_graph_element(graph_element):
  '''
    A learnable graph element is responsible for storing and performing the forward, backward and update operations in a normal neural-network setting.
  '''

  _has_back = False
  _name = 'Undefined'

  def __init__(self, forward_operation, backward_operations = None, previous_elements = None):
    if previous_elements is not None:
      if not isinstance(previous_elements, list):
        previous_elements = [ previous_elements ] 
      for i, prev in enumerate(previous_elements):
        assert isinstance(prev, np.ndarray) or isinstance(prev, learnable_graph_element)
        if isinstance(prev, np.ndarray):
          previous_elements[i] = rm.graph.StaticVariable(prev)
    super().__init__(previous_elements = previous_elements) 
    assert isinstance(forward_operation, operation)
    if backward_operations is None:
      backward_operations = [ ]
    forward_graph =  operational_element(operation = forward_operation, tags = ['Forward'])
    self._fwd = forward_graph
    self._bwd_graphs = [] 
    for op in backward_operations:
      bwd_graph = operational_element(op, tags = ['Backward'])
      self._bwd_graphs.append(bwd_graph)

    for consumed in forward_operation.consumes:
      for op_num, op in enumerate(backward_operations):
        if consumed in op.produces:
          upd = update_operation(0.01, consumer = forward_operation, producer = op, key = consumed)
          upd_g = operational_element(upd, tags = ['Update'])
          upd_g.add_input(self._bwd_graphs[op_num])

    if previous_elements is not None:
      self.connect(previous_elements = previous_elements)
    
    
  def connect(self, previous_elements):

    for elem in previous_elements:
      prev_graph_input = elem.get_forward_output()
      self._fwd.add_input(prev_graph_input)

    for num, elem in enumerate(previous_elements):
      if elem.has_back:
        elem.connect_back(self, pos = num)

    self.forward()

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
    
    def __init__(self, call_list):
      self.call_list = call_list

    def execute(self, epochs = None, steps = 1):
      while(steps > 0):
        try:
          self.perform_step()
          steps -= 1
        except StopIteration:
          raise NotImplementedError()
    
    def perform_step(self):
      for depth in self.call_list.keys():
        for call in self.call_list[depth]:
          call()

    def loss(self):
      return self.loss_func

  def getTrainingExecutor(self):
    self._bwd_graphs[0].setup(tag = 'Update')
    dct = self._bwd_graphs[0].get_call_dict()
    ret = learnable_graph_element.Executor(dct)
    return ret
    

  def forward(self):
    self._fwd.forward()
    return self

  def backward(self):
    if len(self._bwd_graphs[0]._previous_elements) == 0:
      loss = rm.graph.ConstantLoss(previous_element = self)
      loss._bwd_graphs[0].add_input(self._fwd)  
    self._fwd.continue_forward(tag = 'Backward')
    return self

  def get_gradient(self, some_variable):
    assert isinstance(some_variable, rm.graph.core.multi_gpu_variable)
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
    
  def update(self):
    self._fwd.continue_forward(tag = 'Update')

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
