import numpy as np
from add import add, add_back
import abc

class graph_element(abc.ABC):
  '''
    The graph_element class implements a tree-list, composite design that defines a graph.

    Each graph_element has at least one starting point, which determines where the graph starts.

  '''
  
  def __init__(self, previous_elements = []):

    self._start_points = []

    for prev in previous_elements:
      self._start_points.extend(prev._start_points)
    if len(self._start_points) == 0:
      self._start_points.append(self)
    self._previous_elements = previous_elements

    self._next_elements = []

  def __add__(self, other):

    ret = add()
    ret.setup(self._output_signature, other._output_signature)
    # This allows the user to see the result immediately
    ret.perform()

    ret = operational_element(ret, [self, other])
    self._next = ret
    other._next = ret
    return ret 

  @abc.abstractmethod
  def forward(self): pass

  @abc.abstractmethod
  def __repr__(self): pass


class operational_element(graph_element):
  '''
    The operational_element class implements the lowest structure of the graph elements, containing exactly one operation. This operation is perform during the forward call.
  '''
  def __init__(self, operation, previous_elements = []):

    super(operational_element, self).__init__(previous_elements = previous_elements)

    self._op = operation
    self._signature = operation.get_output_signature()
    


  def forward(self):
    for prev in self._previous_elements:
      prev.forward()
    self._op.perform() 


  def __repr__(self):
    raise NotImplementedError('Still figuring out what should represent a graph-element')
