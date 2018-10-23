import numpy as np
import renom as rm

class max_pool:

  def __init__(self, kernel = 1, stride = 1, padding = 1):
    self._kernel = kernel
    self._stride = stride
    self._padding = padding

  def setup(self, inputs):
    num_gpus = len(inputs)
    in_example = inputs[0]
    self._num_gpus = num_gpus
    output_shape = [in_example.shape[0], in_example.shape[1]]
    output_img_shape = [((s + self._padding * 2 - self._kernel) // self._stride + 1) for s in in_example.shape[2:]]
    output_shape.extend(output_img_shape)
    self._inputs = inputs
    self._outputs = []
    self._pool_desc = rm.cuda.PoolingDescriptor((self._kernel,self._kernel), (self._padding,self._padding), (self._stride,self._stride), pool_mode=0)
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu):
        self._outputs.append(rm.GPUValue(shape=output_shape))

  def setup_backward(self, inputs):
    self._backwards = []
    self._inputs_back = inputs
    input_shape = self._inputs[0].shape
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu):
        self._backwards.append(rm.GPUValue(shape=input_shape))
    

  def forward(self):
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu) as handle:
        rm.cuda.cuPoolingForward(handle, self._pool_desc, self._inputs[gpu], self._outputs[gpu])

  def backward(self):
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu) as handle:
        rm.cuda.cuPoolingBackward(handle, self._pool_desc, self._inputs[gpu], self._outputs[gpu], self._inputs_back[gpu], self._backwards[gpu])


  def update(self, lr): pass
