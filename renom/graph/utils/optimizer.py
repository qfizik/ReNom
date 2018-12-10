import renom as rm
import numpy as np
from renom.graph.core import multi_gpu_variable
import renom.utility.initializer as init

class optimizer_factory:

  def __init__(self):
    self._ops = { }
    self.args = ( )
    self.kwargs = { }

  def get_op(self, out):
    if id(out) not in self._ops:
      self._ops[id(out)] = self.create_op()
    return self._ops[id(out)]

  def create_op(self):
    if rm.is_cuda_active():
      ret = self.gpu_op(*self.args, **self.kwargs)
    else:
      ret = self.cpu_op(*self.args, **self.kwargs)
    return ret
    


class sgd_update(optimizer_factory):

  class gpu_op:
  
    def __init__(self, learning_rate, momentum):
      self.learning_rate = learning_rate
      self.momentum = momentum
      self._outputs = None
  
    def setup(self, grad, val):
      if val is self._outputs: return
      self.gpus = grad.gpus
      self._dy = grad
      self._outputs = val
      self._run_avg = multi_gpu_variable(shape = grad.shape, gpus = grad.gpus, initializer = init.Constant(0))

    def update(self):  
      for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
        rm.cuda.cu_optimizer_sgd(self.learning_rate, self.momentum, self._dy[gpu], self._run_avg[gpu], self._outputs[gpu], handle)

  class cpu_op(gpu_op):
  
    def update(self):
      dy = self._dy['cpu']
      cur = self._outputs['cpu']
      avg = self._run_avg['cpu']
      ret = (self.learning_rate * dy + avg * self.momentum)
      self._run_avg['cpu'] = ret
      self._outputs['cpu'] = cur - ret



  def __init__(self, learning_rate = 0.01, momentum = 0.4):
    super().__init__()
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.args = (learning_rate, momentum)



class adagrad_update(optimizer_factory):

  class gpu_op:

    def __init__(self, learning_rate, epsilon):
      self.learning_rate = learning_rate
      self.epsilon = epsilon
      self._outputs = None

    def setup(self, grad, val):
      if val is self._outputs: return
      self.gpus = grad.gpus
      self._dy = grad
      self._outputs = val
      self._prev = multi_gpu_variable(shape = grad.shape, gpus = grad.gpus, initializer = init.Constant(0))

    def update(self):
      for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
        rm.cuda.cu_optimizer_adagrad(self.learning_rate, self.epsilon, self._dy[gpu], self._prev[gpu], self._outputs[gpu], self._prev[gpu])
  
   
  class cpu_op(gpu_op):
    
    def update(self):
      dy = self._dy['cpu']
      cur = self._outputs['cpu']
      pdy = self._prev['cpu']
      r = pdy + dy * dy
      ret = self.learning_rate * dy / (np.sqrt(r) + self.epsilon)
      self._outputs['cpu'] = cur - ret
      self._prev['cpu'] = r


  def __init__(self, learning_rate = 0.01, epsilon = 1e-8):
    super().__init__()
    self._lr = learning_rate
    self._eps = epsilon
    self.args = (learning_rate, epsilon)

class adadelta_update(optimizer_factory):

  class gpu_op:

    def __init__(self, learning_rate, epsilon):
      self.learning_rate = learning_rate
      self.epsilon = epsilon
      self._outputs = None

    def setup(self, grad, val):
      if val is self._outputs: return
      self.gpus = grad.gpus
      self._dy = grad
      self._outputs = val
      self._prev = multi_gpu_variable(shape = grad.shape, gpus = grad.gpus, initializer = init.Constant(0))

    def update(self):
      for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
        rm.cuda.cu_optimizer_adagrad(self.learning_rate, self.epsilon, self._dy[gpu], self._prev[gpu], self._outputs[gpu], self._prev[gpu])
  
   
  class cpu_op(gpu_op):
    
    def update(self):
      dy = self._dy['cpu']
      cur = self._outputs['cpu']
      pdy = self._prev['cpu']
      r = pdy + dy * dy
      ret = self.learning_rate * dy / (np.sqrt(r) + self.epsilon)
      self._outputs['cpu'] = cur - ret
      self._prev['cpu'] = r


  def __init__(self, learning_rate = 0.01, epsilon = 1e-8):
    super().__init__()
    self._lr = learning_rate
    self._eps = epsilon
    self.args = (learning_rate, epsilon)


