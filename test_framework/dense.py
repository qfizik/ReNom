import numpy as np
import renom as rm
from new_gpu import multi_gpu_variable
import renom.utility.initializer as init

class dense:

  def __init__(self, out_size = 1):
    self._out_size = out_size

  def setup(self, inputs):
    reshaped_inputs = []
    self._original_input_shape = inputs[0].shape
    for input in range(len(inputs)):
      reshaped_inputs.append(inputs[input].reshape(inputs[input].shape[0], -1))
    inputs = reshaped_inputs
    num_gpus = len(inputs)
    in_example = inputs[0]
    #self._memory_manager = bla bla fix this
    #self._launch_manager = something
    self._num_gpus = num_gpus
    output_shape = [in_example.shape[0], self._out_size]
    weight_shape = [in_example.shape[1], self._out_size]
    bias_shape = [1, self._out_size]
    self._init = init.GlorotNormal()
    self._weight_shape = weight_shape
    self._bias_shape = bias_shape
    self._input_shape = inputs[0].shape
    self._inputs = inputs
    self._outputs = []
    self._weights = multi_gpu_variable(shape=weight_shape, gpus=num_gpus, allocate_backward=True, initializer = self._init)
    self._biases= multi_gpu_variable(shape=bias_shape, gpus=num_gpus, allocate_backward=True, initializer = self._init)
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu):
        self._outputs.append(rm.GPUValue(shape=output_shape))

  def setup_backward(self, inputs):
    self._backwards = []
    self._backwards_reshape = []
    self._inputs_back = inputs
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu):
        self._backwards_reshape.append(rm.GPUValue(shape=self._input_shape)) 
        self._backwards.append(self._backwards_reshape[gpu].reshape(self._original_input_shape))
    
   
  def forward(self):
    #for handle in self._launch_manager.plan()
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu) as handle:
        rm.cuda.cublas_gemm(self._inputs[gpu], 0, self._wf[gpu], 0, self._outputs[gpu], handle)

  def backward(self):
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu) as handle:
        rm.cuda.cublas_gemm(self._inputs_back[gpu], 0, self._weights.get_forwards(gpu), 1, self._backwards_reshape[gpu], handle)
        rm.cuda.cublas_gemm(self._inputs[gpu], 1, self._inputs_back[gpu], 0, self._weights.get_backwards(gpu), handle)

  def update(self, lr):
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu) as handle:
        rm.cuda.cu_optimizer_sgd(lr, 0, self._weights.get_backwards(gpu), None, self._weights[gpu], handle)
        rm.cuda.cu_optimizer_sgd(lr, 0, self._biases.get_backwards(gpu), None, self._biases[gpu], handle)
    
