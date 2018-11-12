from .graph_element import graph_element, operational_element
from .update_graph import update_operation
import renom as rm
import numpy as np

class learnable_graph_element(graph_element):
  '''
    A learnable graph element is responsible for storing and performing the forward, backward and update operations in a normal neural-network setting.
  '''

  _has_back = False
  _name = 'Undefined'
  _backward_operations = []

  def __init__(self, *args, **kwargs):
    if len(self._forward_operations) > 0:
      assert len(self._forward_operations) == 1
      forward_operation = self._forward_operations[0]
      forward_graph =  operational_element(operation = forward_operation, tags = ['Forward'])
      self._fwd_op = self._forward_operations[0]
      self._fwd = forward_graph
    self._bwd_graphs = [] 
    for op in self._backward_operations:
      bwd_graph = operational_element(op, tags = ['Backward'])
      self._bwd_graphs.append(bwd_graph)
    self.call_lists = {}

    if len(self._forward_operations) > 0:
      for consumed in self._fwd_op.consumes:
        for op_num, op in enumerate(self._backward_operations):
          if consumed in op.produces:
            upd = update_operation(0.01, consumer = self._fwd_op, producer = op, key = consumed)
            upd_g = operational_element(upd, tags = ['Update'])
            upd_g.add_input(self._bwd_graphs[op_num])
    super().__init__(*args, **kwargs) 
    
    
  def connect(self, *previous_elements):

    prvs = []
    should_run = True 
    for num, elem in enumerate(previous_elements):
      if isinstance(elem, np.ndarray):
        should_run = True
        if len(self._previous_elements) > num and isinstance(self._previous_elements[num], rm.graph.StaticVariableElement):
          self._previous_elements[num].value = elem
          elem = self._previous_elements[num]
        else:
          elem = rm.graph.StaticVariableElement(elem)
      else:
        assert isinstance(elem, learnable_graph_element)
      prvs.append(elem)
    previous_elements = prvs

    for elem in previous_elements:
      self.add_input(elem)
      prev_graph_input = elem.get_forward_output()
      self._fwd.add_input(prev_graph_input)


    for num, elem in enumerate(previous_elements):
      if elem.has_back:
        elem.connect_back(self, pos = num)


    if should_run is True:
      self.forward()

    return self

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

  def update(self):
    if 'Update' not in self.call_lists:
      self._bwd_graphs[0].setup(tag = 'Update')
      self.call_lists['Update'] = self._bwd_graphs[0].get_call_dict(tag = 'Update')  
    call_dict = self.call_lists['Update']
    for depth in call_dict.keys():
      for call in call_dict[depth]:
        call()
    return self

  def forward(self):
    if 'Forward' not in self.call_lists:
      self._fwd.setup(tag = 'Forward')
      self.call_lists['Forward'] = self._fwd.get_call_dict(tag = 'Forward')
    call_dict = self.call_lists['Forward']
    for depth in call_dict.keys():
      for call in call_dict[depth]:
        call()
    self.call_lists = {}
    return self

  def backward(self):
    if 'Backward' not in self.call_lists:
      self._bwd_graphs[0].setup(tag = 'Backward')
      self.call_lists['Backward'] = self._bwd_graphs[0].get_call_dict(tag = 'Backward')
    call_dict = self.call_lists['Backward']
    for depth in call_dict.keys():
      for call in call_dict[depth]:
        call()
    self.call_lists = { }
    return self
  #@graph_element.walk_tree
  def print_tree(self):
    #print('I am a {0:s} at depth {1:d}'.format(self.name, self.depth))
    self._fwd.print_tree()

  def get_forward_output(self): return self._fwd
  def get_backward_output(self, num = 0): return self._bwd_graphs[num]

  def as_ndarray(self):
    self.forward()
    return self._fwd.as_ndarray()

  @property
  def weights(self):
    ret = self._fwd._op.get_key('w')
    if ret is None:
      raise AttributeError('{} does not define a weight (w)'.format(self._fwd._op.name))
    return ret

  @property
  def weights_back(self):
    self.backward()
    for grph in self._bwd_graphs:
      ret = grph._op.get_key('w')
      if ret is not None:
        return ret
    raise AttributeError('No backward graph defines a weight (w)'.format(self._fwd._op.name))
 
  @property
  def back(self):
    self.backward()
    for grph in self._bwd_graphs:
      ret = grph._op.get_key('dy')
      if ret is not None:
        return ret
    raise AttributeError('No backward graph defines backward (dy)'.format(self._fwd._op.name))
