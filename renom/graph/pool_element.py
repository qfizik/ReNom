import renom as rm
from .core import operation, learnable_graph_element, operational_element, multi_gpu_variable


class pool_forward(operation):

  name = 'Pool (F)'

  def __init__(self, kernel = 3, padding = 0, stride = 1):
    self._kernel = (kernel, kernel)
    self._padding = (padding, padding)
    self._stride = (stride, stride)

  def setup(self, inputs, storage):

    inputs = inputs[0]['y']
    input_shape = inputs.shape
    self._inputs = inputs
    
    pd = rm.cuda.PoolingDescriptor(self._kernel, self._padding, self._stride, pool_mode = 0)
    self._pool_desc = pd

    imgs = (input_shape[2] + self._padding[0] * 2 - self._kernel[0]) // self._stride[0] + 1
    out_shape = [input_shape[0], input_shape[1], imgs, imgs]
    outs = multi_gpu_variable(shape = out_shape)
    self._outputs = outs
    self._vars = {'y' : outs}

  def perform(self):
    with rm.cuda.RenomHandler() as handle:
      rm.cuda.cuPoolingForward(handle, self._pool_desc, self._inputs[0], self._outputs[0])
    

class pool_backward(operation):

  def __init__(self, associated_forward):
    self._fwd_op = associated_forward

  def setup(self, inputs, storage):
    
    inputs = inputs[0]
    self._inputs = inputs
    out_shape = self._fwd_op._inputs.shape
    self._fwd_in = self._fwd_op._inputs
    self._fwd_out = self._fwd_op._outputs
    outs = multi_gpu_variable(shape = out_shape)
    self._outputs = outs
    

  def perform(self):
    with rm.cuda.RenomHandler() as handle:
      rm.cuda.cuPoolingBackward(handle, self._fwd_op._pool_desc, self._fwd_in[0], self._fwd_out[0], self._inputs[0], self._outputs[0]) 

  def get_output_signature(self): return self._outputs    
 

class MaxPoolElement(learnable_graph_element):

  has_back = True

  def __init__(self, kernel, padding, stride):
    self._krnl = kernel
    self._pad = padding
    self._strd = stride
    fwd_op = pool_forward(kernel, padding, stride)
    self._forward_operations = [ fwd_op ]
    self._backward_operations = [ pool_backward(fwd_op) ]
    super().__init__()
