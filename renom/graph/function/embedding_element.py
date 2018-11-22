from renom.graph.core import learnable_graph_element, operational_element, operation, multi_gpu_variable, GraphFactory, graph_variable
import renom.utility.initializer as init
import renom as rm

class embedding_forward(operation):

  name = 'Embedding (F)'
  consumes = ['w']

  def __init__(self, output_size):
    
    self._output_size = output_size

  def setup(self, inputs, storage):
    weights = inputs[1]['y']
    inputs = inputs[0]['y']
    assert isinstance(inputs, multi_gpu_variable), 'Received {}'.format(type(inputs))
    self.gpus = inputs.gpus
    self._init = init.GlorotNormal()
    self._inputs = inputs
    weight_shape = ( inputs[0].shape[1] , self._output_size )
    weights.__init__( shape = weight_shape , gpus = self.gpus, initializer = self._init)
    output_shape = ( inputs[0].shape[0] , self._output_size )
    outputs = multi_gpu_variable( shape = output_shape, gpus = self.gpus)
    self._vars = {'x' : inputs, 'w' : weights, 'y' : outputs}
    self._weights = weights
    self._outputs = outputs

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cuembedding_forward(self._inputs[gpu], self._weights[gpu], self._outputs[gpu])


class embedding_weight_backward(operation):

  name = 'Embedding Weight (B)'
  produces = ['w']

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward


  def setup(self, inputs, storage):
    inputs = inputs[0]['dy']
    self._inputs = inputs

    gpus = inputs.gpus
    self.gpus = gpus
    fwd_ins = self._fwd_op.get_key('x')
    fwd_weights = self._fwd_op.get_key('w')
    output_shape = fwd_weights.shape

    outputs = multi_gpu_variable(shape = output_shape, gpus = gpus, initializer = None)

    self._vars = { 'y' : outputs, 'w' : outputs , id(fwd_weights) : outputs }

    self._fwd_ins = fwd_ins
    self._outputs = outputs
    
  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cuembedding_backward(self._fwd_ins[gpu], self._inputs[gpu], self._outputs[gpu])

class EmbeddingGraph(learnable_graph_element):

  has_back = True

  def __init__(self, output_size, previous_element = None):
    
    fwd_op = embedding_forward(output_size)
    bwd_ops = [ embedding_weight_backward(associated_forward = fwd_op),]
    self.output_size = output_size

    super().__init__(forward_operation = fwd_op, backward_operations = bwd_ops, previous_elements = previous_element)

class EmbeddingGraphElement(GraphFactory):

  def __init__(self, output_size):    
    super().__init__()
    self.output_size = output_size
    self.params['w'] = graph_variable()
    self._bias = rm.graph.BiasGraphElement()
    self.params['b'] = self._bias.params['b']

  def connect(self, other):
    ret = EmbeddingGraph(output_size = self.output_size, previous_element = [ other, self.params['w']])
    ret = self._bias(ret)
    return ret

