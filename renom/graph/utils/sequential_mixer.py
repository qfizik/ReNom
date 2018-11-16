from renom.graph.core import learnable_graph_element, GraphFactory


class SequentialSubGraph(learnable_graph_element):

  has_back = True

  def __init__(self, graphs):
    assert isinstance(graphs, list)
    self._connect_list(graphs)
    fwd_op = self.first_element._fwd
    super().__init__(forward_operation = fwd_op)
    self._bwd_graphs = self.last_element._bwd_graphs


  def __repr__(self): return self.last.__repr__()
  def get_forward_output(self): return self.last_element._fwd

  def _connect_list(self, graph_list):
    assert isinstance(graph_list, list)
    prev = graph_list[0]
    self.first_element = prev
    assert isinstance(prev, learnable_graph_element)
    for g in range(1, len(graph_list)):
      prev = graph_list[g].connect(prev)
    self.last_element = prev
