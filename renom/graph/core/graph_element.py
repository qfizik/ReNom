import numpy as np
import abc
import functools

class graph_element(abc.ABC):
  '''
    This class implements basic logic of the graph. The graph is constructed as a tree with
    references to both the previous and following elements.

    The basic tree maintains a depth variable with the rule that its depth is the maximum
    of its predecessors plus 1. If updated, all following depth values will be recalculated.

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
    # This should perform a copy, not a reference store.
    self._previous_elements = previous_elements
    self.depth = depth
    self._next_elements = []
    self.update_depth()

  def add_input(self, new_input):
    if new_input not in self._previous_elements:
      self._previous_elements.append(new_input)
    new_input.add_next(self)
    self.update_depth()

  def add_next(self, new_next):
    self._next_elements.append(new_next)

  def remove_input(self, prev_input):
    if prev_input in self._previous_elements:
      prev_input.remove_next(self)
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
      '''
      This method reset all the elements in the
      tree to a non-visited state.
      '''
      if self._visited is False:
        return
      self._visited = False
      for prev in self._previous_elements:
        cleanup(prev)
      for elem in self._next_elements:
        cleanup(elem)

    def walk_func(self, func, dct, *args, **kwargs):
      if self._visited is True: return

      self._visited = True
      assert isinstance(dct, dict)
      for prev in self._previous_elements:
        walk_func(prev, func, dct, *args, **kwargs)

      ret = func(self, *args, **kwargs)
      if ret is not None:
        if self.depth not in dct:
            dct[self.depth] = []
        dct[self.depth].append(ret)

      self._next_elements.sort()
      for elem in self._next_elements:
        if elem.depth == self.depth + 1:
          walk_func(elem, func, dct, *args, **kwargs)


    @functools.wraps(func)
    def ret_func(self, *args, **kwargs):
      ret = {}
      walk_func(self, func, ret, *args, **kwargs)
      cleanup(self)
      return ret
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
  '''
        A utility class designed to work together with the walk_tree function.
        All operational elements will share a reference to the same _vars dictionary
        through the graph_storage class.
  '''
  _vars = { }

  def register(self, key, value):
    if key == 'CallDict':
      self.destroyCallDict()
    self._vars[key] = value

  def retrieve(self, key):
    return self._vars.get(key, None)

  def merge(self, other):
    for key in other._vars.keys():
      self._vars[key] = other._vars[key]
    other._vars = self._vars

  def destroyCallDict(self):
    if 'CallDict' in self._vars and self._vars['CallDict'] is not None:
      dct = self._vars['CallDict']
      for depth in dct:
        for op in range(len(dct[depth])):
          dct[depth][op] = None
      self._vars.pop('CallDict', None)

  '''
    There is currently a bug in Python when attempting to delete function references in a list.
    Manually setting each element to be None prevents the list from trying to do anything with
    with the reference upon deletion.
  '''
  def __del__(self):
    if 'CallDict' in self._vars:
      self.destroyCallDict()
    if 'Updates' in self._vars:
      updts = self._vars['Updates']
      for i in range(len(updts)):
        updts[i] = None
      self._vars.pop('Updates', None)

class operational_element(graph_element):
  '''
        The lowest graph element constructed.

        operational_element requires an operation, which defines what it does. The graph consisting
        of opertional elements is supposed to be constructed and maintained by the algorithms found in
        UserGraph and should not be be constructed directly.
  '''
  def __init__(self, operation, previous_elements = None, tags = None):
    super(operational_element, self).__init__(previous_elements = previous_elements)

    self._storage = graph_storage()
    self.prev_inputs = None
    self._op = operation

    self._tags = []
    if tags is not None:
      self.add_tags(new_tags = tags)
    else: assert False


  def add_input(self, new_input):
    super().add_input(new_input)
    self._storage.merge(new_input._storage)

  def add_tags(self, new_tags):
    for tag in new_tags:
      if tag not in self._tags:
        self._tags.append(tag)


  def check_tags(func):
    @functools.wraps(func)
    def ret_func(self, *args, tag = None, **kwargs):
      if tag in  self._tags or tag is None:
        return func(self, *args, **kwargs)
    return ret_func

  def get_call_dict(self, tag = None):
    self._storage.register('CallDict', None)
    dct = self._storage.retrieve('CallDict')
    assert dct is None
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
    if self._op.perform not in dct[self.depth]:
      dct[self.depth].append(self._op.perform)

  def gather_operations_with_role(self, role, tag = None):
    ret = self._gather_roles(role)
    return ret

  @graph_element.walk_tree
  @check_tags
  def _gather_roles(self, role):
    if role in self._op.roles:
      return self._op

  def gather_operations(self, op, tag = None):
    self._storage.register('Operations', [])
    self._gather_ops(op)
    return self._storage.retrieve('Operations')

  @graph_element.walk_tree
  @check_tags
  def _gather_ops(self, op):
    if isinstance(self._op, op):
      lst = self._storage.retrieve('Operations')
      lst.append(self._op)

  def inputs_changed(self):
    inputs = []
    for prev in self._previous_elements:
      prev_inp = prev.get_output()
      if prev_inp['y'] is None:
        print('{} produced bad output'.format(prev._op.name))
        raise Exception
      inputs.append(prev_inp)
    changed = False
    if self.prev_inputs is not None:
      for inp in inputs:
        if inp not in self.prev_inputs:
          changed = True
          break
    else:
      changed = True
    return changed


  @check_tags
  def forward(self):
    if self.inputs_changed():
      self.setup()
    self._op.perform()


  def finalize(self):
    self.setup_all()
    self._storage.register('Finalized', [])
    finished = False
    while not finished:
      self._smooth_iteration()
      rets = self._storage.retrieve('Finalized')
      finished = all(r is True for r in rets)
    self._finalize()

  @graph_element.walk_tree
  def _smooth_iteration(self):
    lst = self._storage.retrieve('Finalized')
    lst.append(self._op.optimize())


  @graph_element.walk_tree
  def _finalize(self):
    self._op.finalize()




  def calculate_forward(self, tag = None):
    for elem in self._previous_elements:
      elem.calculate_forward(tag)
    self.forward(tag = tag)

  def continue_forward(self, tag = None):
    self.forward(tag = tag)
    for elem in self._next_elements:
      elem.continue_forward(tag)


  def continue_setup(self, tag = None):
    self.setup(tag = tag)
    for elem in self._next_elements:
      elem.continue_setup(tag)

  @check_tags
  def setup(self):
    if not self.inputs_changed():
      return
    inputs = [prev.get_output() for prev in self._previous_elements]
    self._op.setup(inputs)
    self.prev_inputs = inputs

  @graph_element.walk_tree
  def setup_all(self):
    self.setup()

  @graph_element.walk_tree
  def print_tree(self):
    print('I am a {:s} at depth {:d} with tags: {}'.format(self._op.name, self.depth, self._tags))

  @property
  def name(self): return self._op.name

  def add_next(self, new_next):
    assert isinstance(new_next, operational_element)
    super().add_next(new_next)

  @property
  def output(self):
    if self.inputs_changed():
      self.setup()
    ret = self.get_output()['y']
    return ret

  def get_output(self): return self._op.get_output_signature()

  def as_ndarray(self): return self._op.as_ndarray()

  def __repr__(self):
    return self._op.__repr__()
