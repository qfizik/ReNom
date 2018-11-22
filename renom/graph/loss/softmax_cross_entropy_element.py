import renom as rm
from renom.graph.core import operation, loss_graph_element, graph_element, multi_gpu_variable, GraphFactory 

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
      rm.cuda.cumul(tmp, -1, tmp, handle)
      rm.cuda.cudiv(tmp, self._N, tmp, handle)
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


class SoftmaxCrossEntropyElement(loss_graph_element):


  def __init__(self, previous_elements = None):
    fwd_op = softmax_forward()
    bwd_ops = [ softmax_backward(fwd_op)  ]
    super().__init__(forward_operation = fwd_op, backward_operations = bwd_ops, previous_elements = previous_elements)

class SoftmaxCrossEntropyGraphElement(GraphFactory):

  def connect(self, predictions, true_values):
    ret = SoftmaxCrossEntropyElement(previous_elements = [predictions, true_values])
    return ret
