import numpy as np
import renom as rm

class softmax_loss:

  def __init__(self): pass

  def setup(self, inputs):
    num_gpus = len(inputs)
    in_example = inputs[0]
    self._num_gpus = num_gpus
    self._inputs = inputs
    self._outputs = []
    self._input_shape = in_example.shape
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu):
        self._outputs.append(rm.GPUValue(shape=self._input_shape))

  def setup_backward(self, inputs):
    self._backwards = []
    self._inputs_back = inputs
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu):
        self._backwards.append(rm.GPUValue(shape=self._input_shape))


  def forward(self):
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu) as handle:
        rm.cuda.cuSoftmaxForward(handle, self._inputs[gpu], self._outputs[gpu], mode = 1)


  def backward(self):
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu) as handle:
        rm.cuda.cusub(self._outputs[gpu], self._inputs_back[gpu], self._backwards[gpu], handle)
        rm.cuda.cudiv(self._backwards[gpu], self._input_shape[0], self._backwards[gpu], handle)
    
    
  def update(self): pass
