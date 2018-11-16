import renom as rm
from renom.graph.core import operation, learnable_graph_element, multi_gpu_variable, GraphFactory, graph_variable
import renom.utility.initializer as init 
import numpy as np

class lstm_forward(operation):

  name = 'LSTM (F)'
  consumes = ['w','wr','b']
  
  def __init__(self, output_size):
    self._output_size = output_size
    

  def setup(self, inputs, storage):
    bias = inputs[3]['y']
    weight_recursive = inputs[2]['y']
    weight = inputs[1]['y']
    inputs = inputs[0]['y']
    shape = inputs.shape
    gpus = inputs.gpus
    self._current_time = 1
    self.gpus = gpus
    
    w_shape = (inputs.shape[1] , self._output_size * 4)
    wr_shape = (self._output_size, self._output_size * 4)
    b_shape = (1, self._output_size * 4)
    out_shape = (inputs.shape[0], self._output_size)
    tmp_shape = (inputs.shape[0], self._output_size * 4)

    it = init.GlorotNormal()
    #it = init.Constant(1)
    it2 = init.Constant(0)
    weight.__init__(shape = w_shape, gpus = gpus, initializer = it)
    weight_recursive.__init__(shape = wr_shape, gpus = gpus, initializer = it)
    bias.__init__(shape = b_shape, gpus = gpus, initializer = it2)
    outs = multi_gpu_variable(shape = out_shape, gpus = gpus)
    self._prevs = [multi_gpu_variable(shape = out_shape, gpus = gpus, initializer = it2)]
    self._states = [multi_gpu_variable(shape = out_shape, gpus = gpus, initializer = it2)]
    self._tmps = [multi_gpu_variable(shape = tmp_shape, gpus = gpus, initializer = it2)]
    self._ins = [multi_gpu_variable(shape = shape, gpus = gpus, initializer = it2)]

    self._inputs = inputs
    self._outputs = outs
    self._out_shape = out_shape
    self._tmp_shape = tmp_shape

    self._vars = { 'y' : outs, 'w' : weight, 'wr' : weight_recursive, 'b' : bias }
    self._weights = weight
    self._weights_recursive = weight_recursive
    self._bias = bias

  def reset(self):
    self._prevs = [ self._prevs[0] ]
    self._states = [ self._states[0] ]
    self._tmps = [ self._tmps[0] ]
    self._ins = [ self._ins[0] ]
    self._current_time = 1

  def perform(self):
    prev = self._prevs[self._current_time - 1]
    prev_state = self._states[self._current_time - 1]
    tmp_gpus = multi_gpu_variable(shape = self._tmp_shape, gpus = self.gpus)
    tmp_gpus2 = multi_gpu_variable(shape = self._tmp_shape, gpus = self.gpus)
    new_in = multi_gpu_variable(shape = self._inputs.shape, gpus = self.gpus)
    for i, gpu in enumerate(new_in):
      gpu.copy_from(self._inputs[i])
    new_prev = multi_gpu_variable(shape = self._out_shape, gpus = self.gpus)
    new_state = multi_gpu_variable(shape = self._out_shape, gpus = self.gpus)
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      # t1 = dot(x, w)
      rm.cuda.cublas_gemm(self._inputs[gpu], 0, self._weights[gpu], 0, tmp_gpus[gpu], handle)
      # t2 = dot(z, wr)
      rm.cuda.cublas_gemm(prev[gpu], 0, self._weights_recursive[gpu], 0, tmp_gpus2[gpu], handle)
      # t1 = t1 + t2
      rm.cuda.cuadd(tmp_gpus[gpu], tmp_gpus2[gpu], tmp_gpus[gpu], handle)
      # t1 = t1 + b
      rm.cuda.cuadd(tmp_gpus[gpu], self._bias[gpu], tmp_gpus[gpu], handle)
      # t1 = activate(t1)
      rm.cuda.culstm_forward_activate(tmp_gpus[gpu])
      # t1 = lstm_forward(t1, state, prev_state, y)
      rm.cuda.culstm_forward(tmp_gpus[gpu], new_state[gpu], prev_state[gpu], new_prev[gpu])
      self._outputs[gpu].copy_from(new_prev[gpu])
    self._ins.append(new_in)
    self._tmps.append(tmp_gpus)
    self._prevs.append(new_prev)
    self._states.append(new_state)
    self._current_time += 1
      


class lstm_backward(operation):

  name = 'LSTM (B)'
  produces = [ 'w' , 'wr' , 'b' ]

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):
    fwd = self._fwd_op
    inputs = inputs[0]['y']
    gpus = inputs.gpus
    weights = fwd.get_key('w')
    weights_recursive = fwd.get_key('wr')
    bias = fwd.get_key('b')

    self.ini = init.Constant(0)
    weights_back = multi_gpu_variable(shape = weights.shape, gpus = gpus, initializer = self.ini)
    weights_recursive_back = multi_gpu_variable(shape = weights_recursive.shape, gpus = gpus, initializer = self.ini)
    bias_back = multi_gpu_variable(shape = bias.shape, gpus = gpus)
    outs = multi_gpu_variable(shape = fwd._inputs.shape, gpus = gpus, initializer = self.ini)

    self.gpus = gpus
    self._bias = bias
    self._weights = weights
    self._weights_recursive = weights_recursive
    self._weights_back = weights_back
    self._weights_recursive_back = weights_recursive_back
    self._bias_back = bias_back
    self._outputs = outs
    self._inputs = inputs

    self._vars = { 'y' : self._outputs,  'w' : weights_back, 'wr' : weights_recursive_back, 'b' : bias_back, id(fwd._inputs) : self._outputs, id(weights) : weights_back, id(weights_recursive) : weights_recursive_back, id(bias) : bias_back, }
    

  def perform(self):

    self.reset()
    print('Going back')
    time = self._fwd_op._current_time - 1
    tmps = self._fwd_op._tmps
    prevs = self._fwd_op._prevs
    states = self._fwd_op._states
    
    dr_shape = self._fwd_op._tmp_shape
    dou_shape = self._fwd_op._out_shape

    drt = multi_gpu_variable(shape = dr_shape, gpus = self.gpus, initializer = self.ini)
    dou = multi_gpu_variable(shape = dou_shape, gpus = self.gpus, initializer = self.ini)
    pfg = multi_gpu_variable(shape = dr_shape, gpus = self.gpus, initializer = self.ini)

    dr = multi_gpu_variable(shape = dr_shape, gpus = self.gpus, initializer = self.ini)
    dou_n = multi_gpu_variable(shape = dou_shape, gpus = self.gpus, initializer = self.ini)

    dy = multi_gpu_variable(shape = dou_shape, gpus = self.gpus, initializer = self.ini)
    for i, gpu in enumerate(dy):
      gpu.copy_from(self._inputs[i])

    dx = multi_gpu_variable(shape = self._fwd_op._inputs.shape, gpus = self.gpus)

    while (time > 0):
      state = states[time]
      prev_state = states[time - 1]
      tmp = tmps[time]
      for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
        rm.cuda.cutanh( state[gpu] , state[gpu] )
        rm.cuda.culstm_backward( tmp[gpu],
                                 dr[gpu], # Empty like tmps or next dr
                                 state[gpu], # Time or time+1?
                                 prev_state[gpu],
                                 dy[gpu],
                                 pfg[gpu], # tmps for next time
                                 dou[gpu], # outputs
                                 dou_n[gpu] ) # Empty like outputs
        rm.cuda.cublas_gemm(dr[gpu], 0, self._weights[gpu], 1, dx[gpu], handle)
        drt[gpu].copy_from(dr[gpu])
        dou[gpu].copy_from(dou_n[gpu])
        pfg[gpu].copy_from(tmp[gpu])
        rm.cuda.cublas_gemm(dr[gpu], 0, self._weights_recursive[gpu], 1, dy[gpu], handle)
        rm.cuda.cuadd(dx[gpu], self._outputs[gpu], self._outputs[gpu], handle)
      time -= 1                


  def reset(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cusub(self._outputs[gpu], self._outputs[gpu], self._outputs[gpu], handle)



class Lstm(learnable_graph_element):

  has_back = True

  def __init__(self, output_size, previous_element = None):
    self._output_size = output_size
    fwd_op = lstm_forward(output_size)
    bwd_ops = [ lstm_backward(fwd_op) ]
    super().__init__(fwd_op, bwd_ops, previous_element)

  def reset(self):
    self._fwd._op.reset()
    #self._bwd_graphs[0]._op.reset()
  
class LstmElement(GraphFactory):

  def __init__(self, output_size):
    self._output_size = output_size
    self._weights = graph_variable()
    self._weights_r = graph_variable()
    self._bias = graph_variable()

  def connect(self, other):
    return Lstm(self._output_size, [ other, self._weights, self._weights_r, self._bias ])
