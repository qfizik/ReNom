import numpy as np
from add import add, add_back
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
    for next_element in self._next_elements:
      next_element.print_tree()

  @abc.abstractmethod
  def forward(self): pass

  @abc.abstractmethod
  def __repr__(self): pass


class operational_element(graph_element):
  '''
    The operational_element class implements the lowest structure of the graph elements, containing exactly one operation. This operation is perform during the forward call.
  '''
  def __init__(self, operation, previous_elements = None, tags = None):
    super(operational_element, self).__init__(previous_elements = previous_elements)

    self._op = operation
    self._called = False
    self._called_setup = False

    self._tags = []
    if tags is not None:
      self.add_tags(new_tags = tags)

  def is_setup(self): return self._called_setup


  def forward(self, tag):
    if tag not in self._tags:
      return
    assert self._called is False
    self._called = True
    self._op.perform()
    for next_elem in self._next_elements:
      next_elem.forward(tag)
    self._called = False

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

  def setup(self, tag = None):
    if tag in self._tags:
      inputs = []
      self._called_setup = True
      for prev in self._previous_elements:
        if prev.is_setup() is False:
          prev.setup(tag)
        inputs.append(prev.get_output())
      self._op.setup(inputs)
      for next_element in self._next_elements:
        next_element.setup(tag = tag)

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

  def get_output(self): return self._op.get_output_signature()

  def __repr__(self):
    return self._op.__repr__()
