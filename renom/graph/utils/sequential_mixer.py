from .core import learnable_graph_element

def _connect_list(graph_list):
  assert isinstance(graph_list, list)
  for g in range(1, len(graph_list)):
    graph_list[g].connect(graph_list[g-1])

class SequentialSubGraph(learnable_graph_element):

  has_back = True

  def __init__(self, graphs):
    assert isinstance(graphs, list)
    self._forward_operations = []
    self._backward_operations = []
    super().__init__()
    _connect_list(graphs)
    first_element = graphs[0]
    last_element = graphs[len(graphs) - 1]
    self._fwd = first_element._fwd
    self._fwd_op = None
    self._fwd_in = first_element._fwd
    self._bwd_graphs = last_element._bwd_graphs
    self.first = first_element
    self.last = last_element


  def __repr__(self): return self.last.__repr__()
  def get_forward_output(self): return self.last._fwd
