import numpy as np
import renom as rm
rm.set_cuda_active()
#np.set_printoptions(precision = 2, suppress = True)
import pytest
import test_utility

def compare(nd_value, ad_value):
  print('nd=')
  print(nd_value)
  print('ad=')
  print(ad_value)
  assert np.allclose(nd_value, ad_value, atol = 1e-3, rtol = 1e-5)

def onehot(shape):
    N = shape[0]
    D = shape[1]
    ret = np.zeros(shape, dtype=np.float64)
    if D > 1:
        for n in range(N):
            r = np.random.randint(0, D)
            ret[n, r] = 1.
    else:
        ret[np.random.randint(0, N)] = 1
    return ret


def getNumericalDiff( lossMethod, testValue ):
  assert isinstance(testValue, rm.graph.core.multi_gpu_variable)
  coefficients1 = [ 1, -8, 8, -1 ]
  coefficients2 = [ -2, -1, 1, 2 ]
  c = 12
  eps = np.sqrt(np.finfo(rm.precision).eps)

  def store_value(index, storage, value):
    v = storage[0]
    tmp = np.empty(v.shape, dtype = rm.precision)
    v.to_cpu(tmp)
    tmp[index] += value
    v.to_gpu(tmp)

  def retrieve_value(index, storage):
    v = storage[0]
    tmp = np.empty(v.shape, dtype = rm.precision)
    v.to_cpu(tmp)
    return tmp[index]


  diff = np.zeros(testValue.shape)
  for nindex in np.ndindex(diff.shape):
    loss = 0
    for i in range(len(coefficients1)):
      k = retrieve_value(nindex, testValue)
      dx = eps * k if k != 0 else eps
      store_value(nindex, testValue, coefficients2[i] * dx)
      ret = lossMethod() * coefficients1[i]
      store_value(nindex, testValue, -coefficients2[i] * dx)
      loss += ret

    v = loss / (dx * c)
    diff[nindex] = v
  return diff

@test_utility.skipgpu
def test_dense():

  v = np.random.rand(3,4)
  val = rm.graph.StaticVariable(v)
  model = rm.graph.DenseGraphElement(output_size = 2)
  l = rm.graph.ConstantLossElement()
  m = model(val)
  loss = l(m)

  ad = loss.backward().get_gradient(val.value).as_ndarray()
  def func():
    val.forward()
    ret = loss.as_ndarray()
    return ret
  
  compare( getNumericalDiff( func , val.value ) , ad )
  compare( getNumericalDiff( func , model.params['w'].output) , loss.backward().get_gradient(model.params['w'].output).as_ndarray())
  compare( getNumericalDiff( func , model.params['b'].output) , loss.backward().get_gradient(model.params['b'].output).as_ndarray())

@test_utility.skipgpu
def test_conv():

  v = np.random.rand(1,1,5,5)
  val = rm.graph.StaticVariable(v)
  model = rm.graph.ConvolutionalGraphElement()
  loss = rm.graph.ConstantLossElement()
  m = model(val)
  l = loss(m)

  def func():
    m.forward()
    l.forward()
    ret = l.as_ndarray()
    return ret

  compare( getNumericalDiff( func , val.value ) ,  l.backward().get_gradient(val.value).as_ndarray() )
  compare( getNumericalDiff( func , model.params['w'].output ) ,  l.backward().get_gradient(model.params['w'].output).as_ndarray() )
  compare( getNumericalDiff( func , model.params['b'].output ) ,  l.backward().get_gradient(model.params['b'].output).as_ndarray() )


@test_utility.skipgpu
def test_pool():
  v = np.random.rand(1,1,5,5)
  val = rm.graph.StaticVariable(v)
  model = rm.graph.MaxPoolElement(kernel = 3, padding = 0, stride = 1)
  loss = rm.graph.ConstantLossElement()
  m = model(val)
  l = loss(m)

  def func():
    m.forward()
    l.forward()
    ret = l.as_ndarray()
    return ret

  compare( getNumericalDiff( func , val.value ) ,  l.backward().get_gradient(val.value).as_ndarray() )


@test_utility.skipgpu
def test_softmax():
  v = np.random.rand(1,3)
  v2 = onehot((1,3))
  val = rm.graph.StaticVariable(v)
  val2 = rm.graph.StaticVariable(v2)
  model = rm.graph.SoftmaxElement()
  m = model(val, val2)

  def func():
    m.forward()
    ret = m.as_ndarray()
    return ret

  compare( getNumericalDiff( func , val.value ) ,  m.backward().get_gradient(val.value).as_ndarray() )


@test_utility.skipgpu
def test_softmaxGraph():
  v = np.random.rand(1,3)
  val = rm.graph.StaticVariable(v)
  model = rm.graph.SoftmaxGraphElement()
  loss = rm.graph.ConstantLossElement()
  m = model(val)
  l = loss(m)

  def func():
    m.forward()
    l.forward()
    ret = l.as_ndarray()
    return ret

  compare( getNumericalDiff( func , val.value ) ,  l.backward().get_gradient(val.value).as_ndarray() )


@test_utility.skipgpu
def test_relu():
  v = np.random.rand(1,3)
  val = rm.graph.StaticVariable(v)
  model = rm.graph.ReluGraphElement()
  loss = rm.graph.ConstantLossElement()
  m = model(val)
  l = loss(m)

  def func():
    m.forward()
    l.forward()
    ret = l.as_ndarray()
    return ret

  compare( getNumericalDiff( func , val.value ) ,  l.backward().get_gradient(val.value).as_ndarray() )

@test_utility.skipgpu
def test_dropout():
  v = np.random.rand(3,4)
  val = rm.graph.StaticVariable(v)
  model = rm.graph.DropoutGraphElement()
  loss = rm.graph.ConstantLossElement()
  m = model(val)
  l = loss(m)

  def func():
    rm.cuda.curand_generator().set_seed(15)
    m.forward()
    l.forward()
    ret = l.as_ndarray()
    return ret

  rm.cuda.curand_generator().set_seed(15)
  compare( getNumericalDiff( func , val.value ) ,  l.backward().get_gradient(val.value).as_ndarray() )


@test_utility.skipgpu
def test_mean_squared():
  v = np.random.rand(1,3)
  v2 = np.random.rand(1,3)
  val = rm.graph.StaticVariable(v)
  val2 = rm.graph.StaticVariable(v2)
  model = rm.graph.MeanSquaredElement()
  m = model(val, val2)

  def func():
    m.forward()
    ret = m.as_ndarray()
    return ret

  compare( getNumericalDiff( func , val.value ) ,  m.backward().get_gradient(val.value).as_ndarray() )

@test_utility.skipgpu
def test_lstm():
  v = np.random.rand(2,3)
  val = rm.graph.StaticVariable(v)
  model = rm.graph.LstmElement(output_size = 4)
  loss = rm.graph.ConstantLossElement()
  m = model(val)
  l = loss(m)

  def func():
    m.reset()
    m.forward()
    m.forward()
    m.forward()
    l.forward()
    ret = l.as_ndarray()
    return ret
  m.forward()
  m.forward()
  m.forward()
  compare( getNumericalDiff( func , val.value ) ,  l.backward().get_gradient(val.value).as_ndarray() )


@test_utility.skipgpu
def test_batch_norm():
  v = np.random.rand(2,3)
  val = rm.graph.StaticVariable(v)
  m1 = rm.graph.DenseGraphElement(output_size = 3)
  model = rm.graph.BatchNormalizeElement()
  loss = rm.graph.ConstantLossElement()
  m2 = m1(val)
  m = model(m2)
  l = loss(m)

  def func():
    m2.forward()
    m.forward()
    l.forward()
    ret = l.as_ndarray()
    return ret

  compare( getNumericalDiff( func , val.value ) ,  l.backward().get_gradient(val.value).as_ndarray() )
  compare( getNumericalDiff( func , model.weights) ,  l.backward().get_gradient(model.weights).as_ndarray() )
  compare( getNumericalDiff( func , model.bias) ,  l.backward().get_gradient(model.bias).as_ndarray() )





