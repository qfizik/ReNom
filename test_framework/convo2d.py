import numpy as np
import renom as rm
import renom.utility.initializer as init

class convo2d:

  def __init__(self, channels = 1, kernel = 1, padding = 0, stride = 1):
    self._channels = channels
    self._kernel = kernel
    self._padding = padding
    self._stride = stride

  def setup(self, inputs):
    num_gpus = len(inputs)
    in_example = inputs[0]
    output_shape = [in_example.shape[0], self._channels]
    output_img_shape = [((s + self._padding * 2 - self._kernel) // self._stride + 1) for s in in_example.shape[2:]]
    output_shape.extend(output_img_shape)
    weight_shape = [self._channels, in_example.shape[1], self._kernel, self._kernel]
    bias_shape = [1, self._channels, 1, 1] 
    self._weight_shape = weight_shape
    self._bias_shape = bias_shape
    self._num_gpus = num_gpus
    self._inputs = inputs
    self._init = init.GlorotNormal()
    self._weights = []
    self._biases = []
    self._outputs = []
    self._conv_desc = rm.cuda.ConvolutionDescriptor((self._padding,self._padding), (self._stride,self._stride), (1,1), rm.precision)
    self._filter_desc = rm.cuda.FilterDescriptor(weight_shape, rm.precision)
    self._algorithms = { 
      'forward' : 0,
      'backward' : {
         'data' : 0,
          'filter' : 0,
       },
    }
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu):
        self._weights.append(rm.GPUValue(array=self._init(self._weight_shape)))
        self._biases.append(rm.GPUValue(array=np.zeros(self._bias_shape)))
        self._outputs.append(rm.GPUValue(array=np.ones(output_shape))) 

  def setup_backward(self, inputs):
    self._backwards = []
    self._weights_back = []
    self._biases_back = []
    self._inputs_back = inputs
    input_shape = self._inputs[0].shape
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu):
        self._weights_back.append(rm.GPUValue(shape=self._weight_shape))
        self._biases_back.append(rm.GPUValue(shape=self._bias_shape))
        self._backwards.append(rm.GPUValue(shape=input_shape)) 


  def forward(self):
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu) as handle:
        rm.cuda.cuConvolutionForwardBiasActivation(handle, self._conv_desc, self._filter_desc, self._inputs[gpu], self._weights[gpu], self._outputs[gpu], self._biases[gpu], self._algorithms['forward'])
      

  def backward(self): 
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu) as handle:
        rm.cuda.cuActivationBackward(handle, self._outputs[gpu], self._inputs_back[gpu]) 
        rm.cuda.cuConvolutionBackward(handle, self._conv_desc, self._filter_desc, self._inputs[gpu], self._weights[gpu], self._inputs_back[gpu], self._weights_back[gpu], self._biases_back[gpu], self._backwards[gpu], self._algorithms['backward'])


  def update(self, lr): 
    for gpu in range(self._num_gpus):
      with rm.cuda.RenomHandler(gpu) as handle:
        rm.cuda.cu_optimizer_sgd(lr, 0, self._weights_back[gpu], None, self._weights[gpu], handle)
        rm.cuda.cu_optimizer_sgd(lr, 0, self._biases_back[gpu], None, self._biases[gpu], handle)
