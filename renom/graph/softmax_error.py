import renom as rm
from .core import operation, learnable_graph_element, graph_element, multi_gpu_variable 

class softmax_forward(operation):

  name = 'Softmax (F)'

  def setup(self, inputs, storage): pass

  def perform(self): pass

  def get_output_signature(self): return None

  def __repr__(self): return None

class softmax_backward(operation):

  name = 'Softmax (B)'

  def setup(self, inputs, storage):

    predictions = inputs[0]['y']
    labels = inputs[1]['y']
    for a, b in zip(predictions.shape, labels.shape):
      assert a == b, '{} / {}'.format(a, b)
    self._N = predictions.shape[0]
    self._graph_input = predictions
    self._label_input = labels

    gpus = predictions._num_gpus
    self._num_gpus = gpus
    output = multi_gpu_variable(shape = predictions.shape, gpus = gpus)

    self._outputs = output
    self._vars = { 'y' : output ,'dy' : output }

  def perform(self):
    for gpu, handle in enumerate(rm.cuda.RenomHandlers(self._num_gpus)):
      rm.cuda.cuSoftmaxForward(handle, self._graph_input[gpu], self._outputs[gpu], mode = 1)
      rm.cuda.cusub(self._outputs[gpu], self._label_input[gpu], self._outputs[gpu], handle)
      rm.cuda.cudiv(self._outputs[gpu], self._N, self._outputs[gpu], handle)


class SoftmaxElement(learnable_graph_element):

  is_connector_element = True

  def __init__(self, previous_element = None):
    self._forward_operations = [ softmax_forward()  ]
    self._backward_operations = [ softmax_backward()  ]

    super().__init__(previous_elements = previous_element)
    self._calls = None

  def connect(self, *previous_elements):
    super().connect(*previous_elements)
    for elem in previous_elements:
      prev_graph_input = elem.get_forward_output()
      self._bwd_graphs[0].add_input(prev_graph_input)

  def connect_back(self, previous_element): assert False

  def update(self):
    if self._calls is None:
      self._bwd_graphs[0].setup(tag = 'Update')
      self._calls = self._bwd_graphs[0].get_call_dict(tag = 'Update')
      
    for depth in self._calls:
      for call in self._calls[depth]:
        call()

  def forward(self): pass

  def __repr__(self): return self._bwd.__repr__()

