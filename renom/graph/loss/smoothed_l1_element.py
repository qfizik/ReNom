import renom as rm
from renom.graph.core import loss_graph_element, operation, multi_gpu_variable, GraphFactory

class smoothed_l1_forward(operation):

  name = 'Mean Squared (F)'

  def setup(self, inputs, storage):
    predictions = inputs[0]['y']
    real_values = inputs[1]['y']
    self.gpus = predictions.gpus
    self._graph_input = predictions
    self._label_input = real_values

    out_shape = ( 1, )
    self._N = predictions.shape[0]
    assert predictions.shape == real_values.shape
    tmp = multi_gpu_variable(shape = predictions.shape, gpus = self.gpus)
    output = multi_gpu_variable(shape = out_shape, gpus = predictions.gpus)

    self._vars = { 'y' : output }
    self._outputs = output
    self._N = predictions.shape[0]
    self._tmp = tmp

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cusub(self._graph_input[gpu], self._label_input[gpu], self._tmp[gpu], handle)
      rm.cuda.cumul(self._tmp[gpu], self._tmp[gpu], self._tmp[gpu], handle)
      tmp = rm.cu.cusum(self._tmp[gpu], handle)
      rm.cuda.cudiv(tmp, self._N, tmp, handle)
      self._outputs[gpu].copy_from(tmp)


class smoothed_l1_backward(operation):

  name = 'Mean Squared (B)'

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):
  
    predictions = inputs[0]['y']
    real_values = inputs[1]['y']
    self._graph_input = predictions
    self._label_input = real_values
    gpus = predictions.gpus
    self.gpus = gpus
    output = multi_gpu_variable(shape = predictions.shape, gpus = gpus)
    self._outputs = output
    self._vars = { 'y' : output, 'dy' : output, id(self._fwd_op._graph_input) : output }
    self._N = predictions.shape[0]
    
  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cudiv(self._outputs[gpu], self._N, self._outputs[gpu], handle)


class SmoothedL1Element(loss_graph_element):

  def __init__(self, previous_elements = None):

    fwd_op = smoothed_l1_forward()
    bwd_ops = [ smoothed_l1_backward(fwd_op) ] 
    super().__init__(forward_operation = fwd_op, backward_operations = bwd_ops, previous_elements = previous_elements)


class SmoothL1GraphElement(GraphFactory):

  def connect(self, predictions, true_values):
    ret = SmoothedL1Element(previous_elements = [predictions, true_values])
    return ret

