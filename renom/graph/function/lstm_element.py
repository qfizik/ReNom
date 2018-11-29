import renom as rm
from renom.graph.core import learnable_graph_element, operation, GraphFactory, graph_variable, multi_gpu_variable
import numpy as np
import renom.utility.initializer as init

class state:

  def __init__(self, null_state):
    self._null_state = null_state

class lstm_forward(operation):

  def __init__(self, output_size):
    self._output_size = output_size

  def setup(self, inputs, storage):
    weight_r = inputs[2]['y']
    weights = inputs[1]['y']
    inputs = inputs[0]['y']
    gpus = inputs.gpus
    self.gpus = gpus
    self._inputs = inputs
    input_shape = inputs.shape

    weight_shape = (input_shape[0], output_size * 4)
    weight_r_shape = (output_size, output_size * 4)
    out_shape = (input_shape[0], output_size)
    it = init.GlorotNormal()

    weights.__init__(shape = weight_shape, gpus = self.gpus, initializer = it)
    weights_r.__init__(shape = weight_r_shape, gpus = self.gpus, initializer = it)
    outs = multi_gpu_variable(shape = out_shape, gpus = self.gpus)
    self._vars = { 'y' : outs, 'w' : weights, 'wr' : weights_r}
    self._weights = weights
    self._weights_r = weights_r
    self._outputs = outs


  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      dx = self._inputs[gpu]
      y = self._outputs[gpu]
      pass

class lstm_forward_cpu(lstm):

  def perform(self):
    x = self._inputs['cpu']
    w = self._weights['cpu']
    wr = self._weights_r['cpu']
    s = self._state.get_prev('s')
    z = self._state.get_prev('z')
    s = np.zeros((x.shape[0], w.shape[1] // 4), dtype=precision) if ps is None else ps
    z = np.zeros((x.shape[0], w.shape[1] // 4), dtype=precision) if pz is None else pz

    u = np.dot(x, w) + np.dot(z, wr)
    m = u.shape[1] // 4
    u, gated = np.split(u, [m, ], axis=1)
    u = np.tanh(u)

    gated = sigmoid(gated)

    state = gated[:, m:m * 2] * u + gated[:, :m] * s
    ret = tanh(state) * gated[:, m * 2:]

    # Calculate ret
    self._outputs['cpu'] = ret

class lstm_backward(operation):

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):
    inputs = inputs[0]['y']
    gpus = inputs.gpus
    self.gpus = gpus
    self._inputs = inputs

  def perform(self):
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      pass

class lstm_backward_cpu(lstm):

  def perform(self):
    dy = self._inputs['cpu']
    n, m = dy.shape

    w = self.attrs._w
    wr = self.attrs._wr
    b = self.attrs._b

    u = self.attrs._u
    s = tanh(self.attrs._state)

    gated = self.attrs._gated
    gd = gate_diff(gated)
    ps = self.attrs._pstate

    drt = context.restore(wr, np.zeros((n, m * 4), dtype=dy.dtype))
    dou = context.restore(w, np.zeros((n, m), dtype=dy.dtype))

    pfg = self.attrs.get("_pfgate", np.zeros_like(self))

    e = dy

    do = e * s * gd[:, 2 * m:]
    dou = e * gated[:, 2 * m:] * activation_diff(s) + pfg * dou

    df = dou * gd[:, :m] * ps if ps is not None else np.zeros_like(dou)
    di = dou * gd[:, m:2 * m] * u
    dc = dou * activation_diff(u) * gated[:, m:2 * m]

    dr = np.hstack((dc, df, di, do))
    dx = np.dot(dr, w.T)

    context.store(wr, dr)
    context.store(w, dou)

    dw = np.dot(x.T, dr)

    dwr = np.dot(y.T, drt)

    dpz = np.dot(dr, wr.T)

    # Calculate ret
    self._outputs['cpu'] = ret

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

