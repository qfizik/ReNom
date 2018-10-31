import numpy as np
import abc

class graph_element(abc.ABC):
  '''
    The graph_element class implements a tree-list, composite design that defines a graph.

    Each graph_element has at least one starting point, which determines where the graph starts.

  '''
  
  def __init__(self, previous_elements = None):

    if previous_elements is not None:
      print('Creating node {} with children {}'.format(id(self), [id(p) for p in previous_elements]))
    if isinstance(previous_elements, graph_element):
      previous_elements = [ previous_elements ]
    elif previous_elements is None:
      previous_elements = []
    self._start_points = []

    depth = 0
    for prev in previous_elements:
      prev.add_next(self)
    self._previous_elements = previous_elements
    self._depth = depth
    self._next_elements = []
    self.update_depth()
    self.update_start()



  def add_input(self, new_input):

    self._previous_elements.append(new_input)
    new_input.add_next(self)
    self.update_depth()
    self.update_start()

  def add_next(self, new_next):
    self._next_elements.append(new_next)

  def update_start(self):
    starts = [ ]
    for prev in self._previous_elements:
      starts.extend(prev._start_points)
    if len(starts) == 0:
      starts.append(self)
    self._start_points = starts
    for next_element in self._next_elements:
      next_element.update_start()

  def update_depth(self):
    for prev in self._previous_elements:
      if prev._depth >= self._depth:
        self._depth = prev._depth + 1
        for next_element in self._next_elements:
          next_element.update_depth()


  def print_tree(self):
    for start in self._start_points:
      start.print_tree()
    for next_element in self._next_elements:
      next_element.print_tree()

  @abc.abstractmethod
  def forward(self): pass

  @abc.abstractmethod
  def __repr__(self): pass

  def clear(self):
    for elem in self._previous_elements:
      elem.remove_next(self)
    self._previous_elements = []
    self._depth = 0
    self._start_points = []
    for elem in self._next_elements:
      elem.remove_input(self)
    self._next_elements = []

class graph_storage:

  _vars = { }

  def register(self, key, value):
    self._vars[key] = value

  def retrieve(self, key):
    return self._vars[key]

  def merge(self, other):
    for key in other._vars.keys():
      self._vars[key] = other._vars[key]
    other._vars = self._vars

class operational_element(graph_element):
  '''
    The operational_element class implements the lowest structure of the graph elements, containing exactly one operation. This operation is perform during the forward call.
  '''
  def __init__(self, operation, previous_elements = None, tags = None):
    super(operational_element, self).__init__(previous_elements = previous_elements)
  
    self._storage = graph_storage()

    self._op = operation
    self._visited = False 
    self._called = False
    self._called_setup = False

    self._tags = []
    if tags is not None:
      self.add_tags(new_tags = tags)

  def is_setup(self): return self._called_setup

  def add_input(self, new_input):
    super().add_input(new_input)
    self._storage.merge(new_input._storage)

  def visit_once(func):
    def ret_func(self, *args, **kwargs):
      if self._visited:
        return
      self._visited = True
      func(self, *args, **kwargs)
    return ret_func

  def cleanup(self):
    if self._visited is False:
      return
    self._visited = False
    for start in self._start_points:
      start.cleanup()
    for elem in self._next_elements:
      elem.cleanup()

  def forward(self, tag = None):
    if tag is None:
      self._op.perform()
    if tag not in self._tags:
      return
    self._op.perform()
    for next_elem in self._next_elements:
      next_elem.forward(tag)

  def gather_calls(self, call_list, tag):
    if tag not in self._tags:
      return
    assert self.is_setup()
    if self._depth not in call_list:
      call_list[self._depth] = []
    if self._op.perform not in call_list[self._depth]:
      call_list[self._depth].append(self._op.perform)
    for next_elem in self._next_elements:
      next_elem.gather_calls(call_list, tag)

  def add_tags(self, new_tags):
    for tag in new_tags:
      if tag not in self._tags:
        self._tags.append(tag)
        for prev in self._previous_elements:
          prev.add_tags([ tag ])

  @visit_once
  def setup(self, tag = None):
    for start in self._start_points:
      start.setup(tag = tag)
    if tag in self._tags or tag is None:
      inputs = []
      self._called_setup = True
      for prev in self._previous_elements:
        if prev.is_setup() is False:
          prev.setup(tag)
        inputs.append(prev.get_output())
      self._op.setup(inputs, self._storage)
      for next_element in self._next_elements:
        if next_element._depth == self._depth + 1:
          next_element.setup(tag = tag)


  @visit_once
  def print_tree(self):
    print('I am a {:s} at depth {:d}'.format(self._op.name, self._depth))
    #print('My tags are {}'.format(self._tags))
    #print('My starting points are: {}'.format([e._op.name for e in self._start_points]))
    super().print_tree()

  def add_next(self, new_next):
    assert isinstance(new_next, operational_element)
    new_tags = new_next._tags
    self.add_tags(new_tags)
    super().add_next(new_next)

  def get_output(self):# return self._op.get_output_signature()
    ret = self._op.get_output_signature()
    assert ret is not None
    return ret

  def as_ndarray(self): return self._op.as_ndarray()

  def __repr__(self):
    return self._op.__repr__()
