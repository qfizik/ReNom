import renom as rm
import numpy as np
from renom.graph.core import multi_gpu_variable
import renom.utility.initializer as init

class optimizer_factory:

  def __init__(self):
    self._ops = { }

  def get_op(self, op):
    if id(op) not in self._ops:
      self._ops[id(op)] = self.create_op()
    return self._ops[id(op)]

class sgd_update(optimizer_factory):

  def __init__(self, learning_rate = 0.01, momentum = 0.4):
    super().__init__()
    self.learning_rate = learning_rate
    self.momentum = momentum

  def create_op(self):
    if rm.is_cuda_active():
      ret = sgd_update_op(self.learning_rate, self.momentum)
    else:
      ret = sgd_update_op_cpu(self.learning_rate, self.momentum)
    return ret
    

class sgd_update_op:

  def __init__(self, learning_rate, momentum):
    self.learning_rate = learning_rate
    self.momentum = momentum

  def setup(self, grad, val):

    self.gpus = grad.gpus
    self._dy = grad
    self._outputs = val
    self._run_avg = multi_gpu_variable(shape = grad.shape, gpus = grad.gpus, initializer = init.Constant(0))


  def update(self):  
    for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
      rm.cuda.cu_optimizer_sgd(self.learning_rate, self.momentum, self._dy[gpu], self._run_avg[gpu], self._outputs[gpu], handle)

class sgd_update_op_cpu(sgd_update_op):

  def update(self):
    dy = self._dy['cpu']
    cur = self._outputs['cpu']
    avg = self._run_avg['cpu']
    ret = cur - (self.learning_rate * dy + avg * self.momentum)
    self._run_avg['cpu'] = ret
    self._outputs['cpu'] = ret


