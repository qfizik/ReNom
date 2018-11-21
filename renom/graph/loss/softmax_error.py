import renom as rm
from renom.graph.core import operation, learnable_graph_element, graph_element, multi_gpu_variable, GraphFactory 

class softmax_forward(operation):

  name = 'Softmax (F)'

  def setup(self, inputs, storage): 
    assert isinstance(inputs[1], dict)
     
    labels = inputs[1]['y']
    inputs = inputs[0]['y']
    out_shape = ( 1, )
    gpus = inputs.gpus
    act_out = multi_gpu_variable(shape = inputs.shape, gpus = gpus)
    outs = multi_gpu_variable(shape = out_shape, gpus = gpus)
    self.gpus = gpus
    self._outputs = outs
    self._vars = { 'y' : outs }
    self._lbls = labels
    self._act_out = act_out
    self._N = inputs.shape[0]
    self._inputs = inputs

  def perform(self): 
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cuSoftmaxForward(handle, self._inputs[gpu], self._act_out[gpu], mode = 1)
      rm.cuda.cucross_entropy(self._act_out[gpu], self._lbls[gpu], self._act_out[gpu], handle)
      tmp = rm.cuda.cusum(self._act_out[gpu], handle)
      if self._act_out[gpu].shape[0] > 0:
        rm.cuda.cudiv(tmp, -self._N, tmp, handle)
      else:
        rm.cuda.cusub(tmp, tmp, tmp, handle)
      self._outputs[gpu].copy_from(tmp)


class softmax_backward(operation):

  name = 'Softmax (B)'

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):

    predictions = inputs[0]['y']
    labels = inputs[1]['y']
    for a, b in zip(predictions.shape, labels.shape):
      assert a == b, '{} / {}'.format(a, b)
    self._N = predictions.shape[0]
    self._graph_input = predictions
    self._label_input = labels

    gpus = predictions.gpus
    self.gpus = gpus
    output = multi_gpu_variable(shape = predictions.shape, gpus = gpus)

    self._outputs = output
    self._vars = { 'y' : output ,'dy' : output , id(self._fwd_op._inputs) : output}

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cuSoftmaxForward(handle, self._graph_input[gpu], self._outputs[gpu], mode = 1)
      rm.cuda.cusub(self._outputs[gpu], self._label_input[gpu], self._outputs[gpu], handle)
      rm.cuda.cudiv(self._outputs[gpu], self._N, self._outputs[gpu], handle)
      #handle.registerWait()


class SoftmaxElement(learnable_graph_element):

  is_connector_element = True

  def __init__(self, previous_element = None):
    fwd_op = softmax_forward()
    bwd_ops = [ softmax_backward(fwd_op)  ]

    super().__init__(forward_operation = fwd_op, backward_operations = bwd_ops, previous_elements = previous_element)

  def connect(self, *previous_elements):
    previous_elements = list(previous_elements)
    super().connect(previous_elements)
    for elem in previous_elements:
      prev_graph_input = elem.get_forward_output()
      self._bwd_graphs[0].add_input(prev_graph_input)
    self._bwd_graphs[0].add_input(self._fwd)
    return self

  def connect_back(self, previous_element): assert False
  
  @property
  def loss(self):
    self._fwd.setup()
    self._fwd.forward()
    return self._fwd.get_output()['y']

class SoftmaxElemental(GraphFactory):

  def __init__(self): raise NotImplementedError()
