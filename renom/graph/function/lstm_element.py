import renom as rm
from renom.graph.core import learnable_graph_element, operation, GraphFactory, graph_variable, GraphMultiStorage, StateHolder
import numpy as np
import renom.utility.initializer as init


class lstm_forward(operation):

  name = 'LSTM (F)'
  consumes = [ 'w' , 'wr' ]

  def __init__(self, output_size):
    self._output_size = output_size

  def setup(self, inputs, storage):
    weights_r = inputs[2]['y']
    weights = inputs[1]['y']
    inputs = inputs[0]['y']
    gpus = inputs.gpus
    self.gpus = gpus
    self._inputs = inputs
    input_shape = inputs.shape

    weight_shape = (input_shape[1], self._output_size * 4)
    weight_r_shape = (self._output_size, self._output_size * 4)
    out_shape = (input_shape[0], self._output_size)
    it = init.GlorotNormal()
    weights.__init__(shape = weight_shape, gpus = self.gpus, initializer = it)
    weights_r.__init__(shape = weight_r_shape, gpus = self.gpus, initializer = it)
    outs = GraphMultiStorage(shape = out_shape, gpus = self.gpus)
    self._vars = { 'y' : outs, 'w' : weights, 'wr' : weights_r}
    self._weights = weights
    self._weights_r = weights_r
    self._outputs = outs
    self._state = None

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      x = self._inputs[gpu]
      w = self._weights[gpu]
      wr = self._weights_r[gpu]

      if self._state is None:
        shp = (x.shape[0], w.shape[1] // 4)
        self._state = StateHolder()
        for gpu, _ in rm.cuda.RenomHandlers(self.gpus):
          self._state.set_prev('s' + str(gpu), rm.GPUValue(np.zeros(shp)))
          self._state.set_prev('z' + str(gpu), rm.GPUValue(np.zeros(shp)))
          self._state.set_prev('pfgate' + str(gpu), rm.GPUValue(np.zeros(shp)))

      s_p = self._state.get_prev('s' + str(gpu))
      z_p = self._state.get_prev('z' + str(gpu))
      tmp1 = rm.GPUValue(shape=(x.shape[0], w.shape[1]))
      tmp2 = rm.GPUValue(shape=(x.shape[0], w.shape[1]))

      rm.cuda.cublas_gemm(x, 0, w, 0, tmp1, handle)
      rm.cuda.cublas_gemm(z_p, 0, wr, 0, tmp2, handle)

      u = tmp1 + tmp2

      z = z_p.empty_like_me()
      state = s_p.empty_like_me()

      rm.cuda.culstm_forward_activate(u)
      rm.cuda.culstm_forward(u, state, s_p, z)

      ret = z
      self._outputs[gpu] = ret
      if self._state._cur_time > 0:
        self._state.set_prev('pfgate' + str(gpu), u)
      self._state.push({ 'x' + str(gpu) : x,
                         'ps' + str(gpu) : s_p,
                         's' + str(gpu) : state,
                         'z' + str(gpu) : ret,
                         'u' + str(gpu) : u,
                         'y' + str(gpu) : ret,
                       })


  def reset(self):
    self._state = None

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class lstm_forward_cpu(lstm_forward):

  def perform(self):
    x = self._inputs['cpu']
    w = self._weights['cpu']
    wr = self._weights_r['cpu']
    if self._state is None:
      self._state = StateHolder({'s' : np.zeros((x.shape[0], w.shape[1] // 4), dtype=rm.precision),
                           'z' : np.zeros((x.shape[0], w.shape[1] // 4), dtype=rm.precision),
                           'pfgate' : np.zeros((x.shape[0], w.shape[1] // 4)),
                            })
    s = self._state.get_prev('s')
    z = self._state.get_prev('z')

    u = np.dot(x, w) + np.dot(z, wr)
    m = u.shape[1] // 4
    u, gated = np.split(u, [m, ], axis=1)
    u = np.tanh(u)

    gated = sigmoid(gated)

    state = gated[:, m:m * 2] * u + gated[:, :m] * s
    ret = np.tanh(state) * gated[:, m * 2:]

    # Calculate ret
    self._outputs['cpu'] = ret
    if self._state._cur_time > 0:
      self._state.set_prev('pfgate', gated[:, :m])
    self._state.push({ 'x' : x,
                       'ps' : s,
                       's' : state,
                       'z' : ret,
                       'gate' : gated,
                       'u' : u,
                       'y' : ret,
                     })
                       

class lstm_backward(operation):

  name = 'LSTM (B)'
  produces = [ 'w' , 'wr' ]

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):
    inputs = inputs[0]['y']
    gpus = inputs.gpus
    self.gpus = gpus
    self._inputs = inputs
    self._state = None
    outs = GraphMultiStorage(shape = self._fwd_op._inputs.shape, gpus = self.gpus)
    w_out = GraphMultiStorage(shape = self._fwd_op._weights.shape, gpus = self.gpus)
    w_r_out = GraphMultiStorage(shape = self._fwd_op._weights_r.shape, gpus = self.gpus)
    self._outputs = outs
    self._w_out = w_out
    self._w_r_out = w_r_out
    self._vars = { 'y' : outs, id(self._fwd_op._inputs) : outs ,
                   'w' : w_out, id(self._fwd_op._weights) : w_out,
                   'wr' : w_r_out, id(self._fwd_op._weights_r) : w_r_out,
                 }

  def perform(self):
    self._state = self._fwd_op._state
    _t = self._state._cur_time
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      dy = self._inputs[gpu]
      w = self._fwd_op._weights[gpu]
      wr = self._fwd_op._weights_r[gpu]

      self._outputs[gpu] = self._outputs[gpu].zeros_like_me()
      self._w_out[gpu] = self._w_out[gpu].zeros_like_me()
      self._w_r_out[gpu] = self._w_r_out[gpu].zeros_like_me()

      drt = None
      dou = None
      pfg = None
      self._state._cur_time = _t
      while(self._state._cur_time > 0):
        cur_state = self._state.peek()
        x = cur_state['x' + str(gpu)]
        y = cur_state['y' + str(gpu)]

        u = cur_state['u' + str(gpu)]
        s = cur_state['s' + str(gpu)]
        rm.cuda.cutanh(s, s)
        ps = cur_state['ps' + str(gpu)]

        if drt is None:
          drt = u.zeros_like_me()
          dou = dy.zeros_like_me()

        pfg = cur_state['pfgate' + str(gpu)]

        e = dy

        dr, dou_n = (a.empty_like_me() for a in (drt, dou))

         
        rm.cuda.culstm_backward(u, dr, s, ps, e, pfg, dou, dou_n)
        dx = rm.GPUValue(shape = (dr.shape[0], w.shape[0]))
        rm.cuda.cublas_gemm(dr, 0, w, 1, dx, handle)


        dw = rm.GPUValue(shape=(x.shape[1], dr.shape[1]))
        rm.cuda.cublas_gemm(x, 1, dr, 0, dw, handle)
        
        dwr = rm.GPUValue(shape=(y.shape[1], drt.shape[1]))
        rm.cuda.cublas_gemm(y, 1, drt, 0, dwr, handle)


        self._outputs[gpu] += dx
        self._w_out[gpu] += dw 
        self._w_r_out[gpu] += dwr
        
        drt = dr
        dou = dou_n
        self._state._cur_time -= 1
        tmp = dy.empty_like_me()
        rm.cuda.cublas_gemm(dr, 0, wr, 1, tmp, handle)
        dy = tmp

def gate_diff(x):
  return x * (-x + 1.)

def activation_diff(x):
  return (1.0 - x ** 2)

class lstm_backward_cpu(lstm_backward):

  def perform(self):
    self._state = self._fwd_op._state

    dy = self._inputs['cpu']
    n, m = dy.shape
    dx, dw, dwr = (0, 0, 0)

    w = self._fwd_op._weights['cpu']
    wr = self._fwd_op._weights_r['cpu']
    drt = np.zeros((n, m * 4), dtype=dy.dtype)
    dou = np.zeros((n, m), dtype=dy.dtype)

    while(self._state._cur_time > 0):
      cur_state = self._state.peek()
      u = cur_state['u']
      s = np.tanh(cur_state['s'])
      x = cur_state['x']
      y = cur_state['y']

      gated = cur_state['gate']
      gd = gate_diff(gated)
      ps = cur_state['ps']


      pfg = cur_state['pfgate']

      e = dy

      do = e * s * gd[:, 2 * m:]
      dou = e * gated[:, 2 * m:] * activation_diff(s) + pfg * dou

      df = dou * gd[:, :m] * ps
      di = dou * gd[:, m:2 * m] * u
      dc = dou * activation_diff(u) * gated[:, m:2 * m]

      dr = np.hstack((dc, df, di, do))
      dx += np.dot(dr, w.T)

      dw += np.dot(x.T, dr)

      dwr += np.dot(y.T, drt)

      dy = np.dot(dr, wr.T)
      drt = dr
      self._state.pop()

    # Calculate ret
    self._outputs['cpu'] = dx 
    self._w_out['cpu'] = dw
    self._w_r_out['cpu'] = dwr

class LstmElement(learnable_graph_element):

  has_back = True

  def __init__(self, output_size, previous_elements = None):
    fwd_op = lstm_forward(output_size) if rm.is_cuda_active() else lstm_forward_cpu(output_size)
    bwd_ops = [ lstm_backward(fwd_op) if rm.is_cuda_active() else lstm_backward_cpu(fwd_op) ]
    super().__init__(forward_operation = fwd_op, backward_operations = bwd_ops, previous_elements = previous_elements)

class LstmGraphElement(GraphFactory):

  def __init__(self, output_size = 3):
    super().__init__()
    self._output_size = output_size
    self.params['w'] = graph_variable()
    self.params['wr'] = graph_variable()
    self._l = None

  def reset(self):
    if self._l is not None:
      self._l.disconnect()
    self._l = None

  def connect(self, other):
    if self._l is None:
      ret = LstmElement(self._output_size, previous_elements = [other, self.params['w'], self.params['wr']])
    else:
      ret = self._l
      prvs = rm.graph.core.learnable_graph._prepare_prevs([other, self.params['w'], self.params['wr']])
      assert prvs[1].output is self.params['w'].output
      assert prvs[2].output is self.params['wr'].output
      ret.connect(prvs)
    self._l = ret
    return ret

