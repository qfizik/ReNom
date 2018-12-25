from renom.graph.core import UserGraph, GraphFactory
import renom as rm
import numpy as np


class SequentialSubGraph(GraphFactory):

  def __init__(self, graphs):
    super().__init__()
    for lvl in range(len(graphs)):
      setattr(self, 'l{:d}'.format(lvl), graphs[lvl])
    self.graphs = graphs

  def connect(self, other):
    if isinstance(other, np.ndarray):
        other = rm.graph.StaticVariable(other)
    assert isinstance(other, UserGraph)
    ret = self._connect_graphs(other)
    return ret

  def _connect_graphs(self, init):
    prev = init
    for graph in self.graphs:
      prev = graph(prev)
    return prev
