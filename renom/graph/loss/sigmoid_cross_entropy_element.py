import renom as rm
from renom.graph.core import operation, loss_graph_element, graph_element, multi_gpu_variable, GraphFactory 

class sigmoid_forward(operation):

  name = 'Sigmoid (F)'

  def setup(self, inputs, storage): 
    assert isinstance(inputs[1], dict)
     
    labels = inputs[1]['y']
    inputs = inputs[0]['y']
    out_shape = ( 1, )
    gpus = inputs.gpus
    act_out1 = multi_gpu_variable(shape = inputs.shape, gpus = gpus)
    act_out2 = multi_gpu_variable(shape = inputs.shape, gpus = gpus)
    act_out3 = multi_gpu_variable(shape = inputs.shape, gpus = gpus)
    outs = multi_gpu_variable(shape = out_shape, gpus = gpus)
    self.gpus = gpus
    self._outputs = outs
    self._vars = { 'y' : outs }
    self._lbls = labels
    self._act_out1 = act_out1
    self._act_out2 = act_out2
    self._act_out3 = act_out3
    self._N = inputs.shape[0]
    self._inputs = inputs

  def perform(self): 
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      # tmp1 = sigmoid(predictions)
      rm.cuda.cusigmoid(self._inputs[gpu], self._act_out1[gpu])
      # tmp2 = cross_entropy(tmp1, true_values)
      rm.cuda.cucross_entropy(self._act_out1[gpu], self._lbls[gpu], self._act_out2[gpu], handle)
      # tmp1 = tmp1 * -1
      rm.cuda.cumul(self._act_out1[gpu], -1, self._act_out1[gpu], handle)
      # tmp1 = tmp1 + 1
      rm.cuda.cuadd(self._act_out1[gpu], 1, self._act_out1[gpu], handle)
      # tmp3 = true_values * -1
      rm.cuda.cumul(self._lbls[gpu], -1, self._act_out3[gpu], handle)
      # tmp3 = tmp3 + 1
      rm.cuda.cuadd(self._act_out3[gpu], 1, self._act_out3[gpu], handle)
      # tmp3 = cross_entropy(tmp1, tmp3)
      rm.cuda.cucross_entropy(self._act_out1[gpu], self._act_out3[gpu], self._act_out3[gpu], handle)
      # tmp2 = tmp2 + tmp3
      rm.cuda.cuadd(self._act_out2[gpu], self._act_out3[gpu], self._act_out2[gpu], handle)
      # tmp2 = tmp2 * -1
      rm.cuda.cumul(self._act_out2[gpu], -1, self._act_out2[gpu], handle)
      tmp = rm.cuda.cusum(self._act_out2[gpu], handle)
      self._outputs[gpu].copy_from(tmp)
      rm.cuda.cudiv(self._outputs[gpu], self._N, self._outputs[gpu], handle)


class sigmoid_backward(operation):

  name = 'Sigmoid (B)'

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):

    predictions = inputs[0]['y']
    labels = inputs[1]['y']
    self._N = predictions.shape[0]
    self._graph_input = predictions
    self._label_input = labels

    gpus = predictions.gpus
    self.gpus = gpus
    output = multi_gpu_variable(shape = predictions.shape, gpus = gpus)
    act_out1 = multi_gpu_variable(shape = predictions.shape, gpus = gpus)

    self._act_out1 = act_out1
    self._outputs = output
    self._vars = { 'y' : output ,'dy' : output , id(self._fwd_op._inputs) : output}

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cusigmoid(self._graph_input[gpu], self._act_out1[gpu])
      rm.cuda.cusub(self._act_out1[gpu], self._label_input[gpu], self._outputs[gpu], handle)
      rm.cuda.cudiv(self._outputs[gpu], self._N, self._outputs[gpu], handle)


class SigmoidCrossEntropyElement(loss_graph_element):


  def __init__(self, previous_elements = None):
    fwd_op = sigmoid_forward()
    bwd_ops = [ sigmoid_backward(fwd_op)  ]
    super().__init__(forward_operation = fwd_op, backward_operations = bwd_ops, previous_elements = previous_elements)

class SigmoidCrossEntropyGraphElement(GraphFactory):

  def connect(self, predictions, true_values):
    ret = SigmoidCrossEntropyElement(previous_elements = [ predictions, true_values])
    return ret

