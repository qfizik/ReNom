from .graph_element import graph_element, operational_element
from .update_graph import update_operation

class learnable_graph_element(graph_element):
  '''
    A learnable graph element is responsible for storing and performing the forward, backward and update operations in a normal neural-network setting.
  '''

  _has_back = False
  _name = 'Undefined'

  def __init__(self, *args, **kwargs):
    assert len(self._forward_operations) == 1
    forward_operation = self._forward_operations[0]
    forward_graph =  operational_element(operation = forward_operation, tags = ['Forward'])
    self._fwd_op = self._forward_operations[0]
    self._fwd = forward_graph
    super().__init__(*args, **kwargs) 
    
    
  def connect(self, *previous_elements):

    for elem in previous_elements:
      super().add_input(elem)
      prev_graph_input = elem.get_forward_output()
      self._fwd.add_input(prev_graph_input)

    self._bwd_graphs = [] 

    for op in self._backward_operations:
      bwd_graph = operational_element(op, tags = ['Backward'])
      self._bwd_graphs.append(bwd_graph)

    for consumed in self._fwd_op.consumes:
      for op_num, op in enumerate(self._backward_operations):
        if consumed in op.produces:
          upd = update_operation(0.01, consumer = self._fwd_op, producer = op, key = consumed)
          upd_g = operational_element(upd, tags = ['Update'])
          upd_g.add_input(self._bwd_graphs[op_num])


    for num, elem in enumerate(previous_elements):
      if elem.has_back:
        elem.connect_back(self, pos = num)


  def connect_back(self, previous_element, pos = 0):
    backward_graph_input = previous_element.get_backward_output(pos)
    for graph in self._bwd_graphs:
      graph.add_input(backward_graph_input)
    

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

  def forward(self):
    self._fwd.setup(tag = 'Forward')
    self._fwd.forward(tag = 'Forward')

  @graph_element.walk_tree
  def print_tree(self):
    print('I am a {0:s} at depth {1:d}'.format(self.name, self.depth))
  #def print_tree(self): self._fwd.print_tree()

  def get_forward_output(self): return self._fwd
  def get_backward_output(self, num = 0): return self._bwd_graphs[num]

  def as_ndarray(self): return self._fwd.as_ndarray()
