import numpy as np
import abc
import functools

class graph_element(abc.ABC):
  '''
    The graph_element class implements a tree-list, composite design that defines a graph.

    Each graph_element has at least one starting point, which determines where the graph starts.

  '''
  
  def __init__(self, previous_elements = None):

    if isinstance(previous_elements, graph_element):
      previous_elements = [ previous_elements ]
    elif previous_elements is None:
      previous_elements = []

    self._visited = False 
    depth = 0
    for prev in previous_elements:
      prev.add_next(self)
    self._previous_elements = previous_elements
    self.depth = depth
    self._next_elements = []
    self.update_depth()

  def add_input(self, new_input):
    self._previous_elements.append(new_input)
    new_input.add_next(self)
    self.update_depth()

  def add_next(self, new_next):
    self._next_elements.append(new_next)

  def remove_input(self, prev_input):
    if prev_input in self._previous_elements:
      self._previous_elements.remove(prev_input)

  def remove_next(self, prev_next):
    if prev_next in self._next_elements:
      self._next_elements.remove(prev_next)

  def update_depth(self):
    for prev in self._previous_elements:
      if prev.depth >= self.depth:
        self.depth = prev.depth + 1
        for next_element in self._next_elements:
          next_element.update_depth()


  def __lt__(self, other):
    return self.depth < other.depth

  @abc.abstractmethod
  def forward(self): pass

  @abc.abstractmethod
  def __repr__(self): pass

  def walk_tree(func):
    def cleanup(self):
      if self._visited is False:
        return
      self._visited = False
      for prev in self._previous_elements:
        cleanup(prev)
      for elem in self._next_elements:
        cleanup(elem)

    def walk_func(self, func, *args, **kwargs):
      if self._visited is True:
        return
      self._visited = True 
      for prev in self._previous_elements:
        walk_func(prev, func, *args, **kwargs)
      func(self, *args, **kwargs)
      self._next_elements.sort()
      for elem in self._next_elements:
        if elem.depth == self.depth + 1:
          walk_func(elem, func, *args, **kwargs)
    @functools.wraps(func)
    def ret_func(self, *args, **kwargs):
      walk_func(self, func, *args, **kwargs)
      cleanup(self)
    return ret_func


  def clear(self):
    for elem in self._previous_elements:
      elem.remove_next(self)
    self._previous_elements = []
    self.depth = 0
    self._start_points = []
    for elem in self._next_elements:
      elem.remove_input(self)
    self._next_elements = []

class graph_storage:

  _vars = { }

  def register(self, key, value):
    self._vars[key] = value

  def retrieve(self, key):
    return self._vars.get(key, None)

  def merge(self, other):
    for key in other._vars.keys():
      self._vars[key] = other._vars[key]
    other._vars = self._vars


  '''
    There is currently a bug in Python when attempting to delete function references in a list.
    Manually setting each element to be None prevents the list from trying to do anything with
    with the reference upon deletion.
  '''
  def __del__(self):
    if 'CallDict' in self._vars:
      dct = self._vars['CallDict']
      for depth in dct:
        for op in range(len(dct[depth])):
          dct[depth][op] = None

class operational_element(graph_element):
  '''
    The operational_element class implements the lowest structure of the graph elements, containing exactly one operation. This operation is perform during the forward call.
  '''
  def __init__(self, operation, previous_elements = None, tags = None):
    super(operational_element, self).__init__(previous_elements = previous_elements)
  
    self._storage = graph_storage()

    self._op = operation

    self._tags = []
    if tags is not None:
      self.add_tags(new_tags = tags)


  def add_input(self, new_input):
    super().add_input(new_input)
    self._storage.merge(new_input._storage)



  def add_tags(self, new_tags):
    for tag in new_tags:
      if tag not in self._tags:
        self._tags.append(tag)
        for prev in self._previous_elements:
          prev.add_tags([ tag ])


  def check_tags(func):
    @functools.wraps(func)
    def ret_func(self, *args, tag = None, **kwargs):
      if tag in  self._tags or tag is None:
        func(self, *args, **kwargs)
    return ret_func

  def get_call_dict(self, tag):
    dct = self._storage.retrieve('CallDict')
    if dct is None:
      self._storage.register('CallDict', {})
      self._create_call_dict(tag = tag)
      dct = self._storage.retrieve('CallDict')
    return dct

  @graph_element.walk_tree
  @check_tags
  def _create_call_dict(self):
    dct = self._storage.retrieve('CallDict')
    if self.depth not in dct:
      dct[self.depth] = [ ]
    dct[self.depth].append(self._op.perform)

  @graph_element.walk_tree
  @check_tags
  def forward(self):
    self._op.perform()

  @graph_element.walk_tree
  @check_tags
  def setup(self):
    inputs = []
    for prev in self._previous_elements:
      inputs.append(prev.get_output())
    self._op.setup(inputs, self._storage)

  @graph_element.walk_tree
  def print_tree(self): 
    print('I am a {:s} at depth {:d}'.format(self._op.name, self.depth))


  def add_next(self, new_next):
    assert isinstance(new_next, operational_element)
    new_tags = new_next._tags
    self.add_tags(new_tags)
    super().add_next(new_next)

  def get_output(self): return self._op.get_output_signature()

  def as_ndarray(self): return self._op.as_ndarray()

  def __repr__(self):
    return self._op.__repr__()
