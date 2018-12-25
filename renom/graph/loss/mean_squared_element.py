import renom as rm
from renom.graph.core import UserLossGraph, operation, GraphMultiStorage, GraphFactory
import numpy as np

class mean_squared_forward(operation):

  name = 'Mean Squared (F)'
  roles = [ 'loss' ]

  def setup(self, inputs):
    predictions = inputs[0]['y']
    real_values = inputs[1]['y']
    self.gpus = predictions.gpus
    self._graph_input = predictions
    self._label_input = real_values

    out_shape = ( 1, )
    self._N = predictions.shape[0]
    assert predictions.shape == real_values.shape
    tmp = GraphMultiStorage(shape = predictions.shape, gpus = self.gpus)
    output = GraphMultiStorage(shape = out_shape, gpus = predictions.gpus)

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

class mean_squared_forward_cpu(mean_squared_forward):

  def perform(self):
    pred = self._graph_input['cpu']
    real = self._label_input['cpu']
    N = len(pred)
    ret = np.sum((pred - real) ** 2).reshape(1,) / (N * 2)
    self._outputs['cpu'] = ret

class mean_squared_backward(operation):

  name = 'Mean Squared (B)'

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs):
  
    predictions = inputs[0]['y']
    real_values = inputs[1]['y']
    self._graph_input = predictions
    self._label_input = real_values
    gpus = predictions.gpus
    self.gpus = gpus
    output = GraphMultiStorage(shape = predictions.shape, gpus = gpus)
    self._outputs = output
    self._vars = { 'y' : output, 'dy' : output, id(self._fwd_op._graph_input) : output }
    self._N = predictions.shape[0]
    
  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cusub(self._graph_input[gpu], self._label_input[gpu], self._outputs[gpu], handle)
      rm.cuda.cumul(self._outputs[gpu], 2, self._outputs[gpu], handle)
      rm.cuda.cudiv(self._outputs[gpu], self._N, self._outputs[gpu], handle)

class mean_squared_backward_cpu(mean_squared_backward):

  def perform(self):
    #dy = self._inputs['cpu']
    pred = self._graph_input['cpu']
    real = self._label_input['cpu']
    N = len(pred)
    ret = (pred - real) / N
    self._outputs['cpu'] = ret

class MeanSquaredElement(UserLossGraph):

  def __init__(self, previous_elements = None):

    fwd_op = mean_squared_forward() if rm.is_cuda_active() else mean_squared_forward_cpu()
    bwd_ops = [ mean_squared_backward(fwd_op) if rm.is_cuda_active() else mean_squared_backward_cpu(fwd_op) ] 
    super().__init__(forward_operation = fwd_op, backward_operations = bwd_ops, previous_elements = previous_elements)


class MeanSquaredGraphElement(GraphFactory):

  def connect(self, predictions, true_values):
    ret = MeanSquaredElement(previous_elements = [predictions, true_values])
    return ret
