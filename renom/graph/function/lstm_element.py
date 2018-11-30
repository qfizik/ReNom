import renom as rm
from renom.graph.core import learnable_graph_element, operation, GraphFactory, graph_variable, multi_gpu_variable, StateHolder
import numpy as np
import renom.utility.initializer as init


class lstm_forward(operation):

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
    outs = multi_gpu_variable(shape = out_shape, gpus = self.gpus)
    self._vars = { 'y' : outs, 'w' : weights, 'wr' : weights_r}
    self._weights = weights
    self._weights_r = weights_r
    self._outputs = outs
    self._state = None

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      raise NotImplementedError()
      x = self._inputs[gpu]
      w = self._weights[gpu]
      wr = self._weights_r[gpu]
      if self._state is None:
        shp = (x.shape[0], w.shape[1] // 4)
        self._state = StateHolder({'s' : multi_gpu_variable(shape = shp, gpus = self.gpus, initializer = init.Constant(0)),
                           'z' : multi_gpu_variable(shape = shp, gpus = self.gpus, initializer = init.Constant(0)),
                           'pfgate' : multi_gpu_variable(shape = shp, gpus = self.gpus, initializer = init.Constant(0)),
                            })
      s_p = self._state.get_prev('s')[gpu]
      z_p = self._state.get_prev('z')[gpu]
      tmp1 = rm.GPUValue(shape=(x.shape[0], w.shape[1]))
      tmp2 = rm.GPUValue(shape=(x.shape[0], w.shape[1]))

      rm.cuda.cublas_gemm(x, 0, w, 0, tmp1, handle)
      rm.cuda.cublas_gemm(z_p, 0, wr, 0, tmp2, handle)
      u = dot(x, w) + dot(z_p, wr)

      z = get_gpu(z_p).empty_like_me()
      state = get_gpu(s_p).empty_like_me()

      cu.culstm_forward_activate(get_gpu(u))
      cu.culstm_forward(get_gpu(u), get_gpu(state), get_gpu(s_p), get_gpu(z))

      ret = cls._create_node(z)

      ret.attrs._x = x
      ret.attrs._w = w
      ret.attrs._wr = wr
      ret.attrs._b = b
      ret.attrs._pz = pz
      ret.attrs._u = u
      ret.attrs._pstate = s_p
      ret.attrs._state = state
      ret._state = state

      if isinstance(pz, Node):
          pz.attrs._pfgate = u


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

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):
    inputs = inputs[0]['y']
    gpus = inputs.gpus
    self.gpus = gpus
    self._inputs = inputs
    self._state = None
    outs = multi_gpu_variable(shape = self._fwd_op._inputs.shape, gpus = self.gpus)
    w_out = multi_gpu_variable(shape = self._fwd_op._weights.shape, gpus = self.gpus)
    w_r_out = multi_gpu_variable(shape = self._fwd_op._weights_r.shape, gpus = self.gpus)
    self._outputs = outs
    self._w_out = w_out
    self._w_r_out = w_r_out
    self._vars = { 'y' : outs, id(self._fwd_op._inputs) : outs ,
                   'w' : w_out, id(self._fwd_op._weights) : w_out,
                   'wr' : w_r_out, id(self._fwd_op._weights_r) : w_r_out,
                 }

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      pass

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

  def connect(self, other):
    ret = LstmElement(self._output_size, previous_elements = [other, self.params['w'], self.params['wr']])
    return ret

