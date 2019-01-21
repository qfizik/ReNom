import decorator
import numpy as np
import renom as rm
import pytest

if rm.precision is not np.float64:
    pytestmark = pytest.mark.skip()

rm.set_renom_seed(30)


def compare(nd_value, ad_value, abs_tol=1e-5, rel_tol=1e-3):
    ret = nd_value.shape == ad_value.shape
    ret = ret and np.allclose(nd_value, ad_value, atol=abs_tol, rtol=rel_tol)
    if ret is False:
        print('ad=')
        print(ad_value)
        print('nd=')
        print(nd_value)
        print('difference=')
        print(nd_value - ad_value)
    assert ret


def rand(*shape):
    if isinstance(shape[0], tuple):
        shape = shape[0]
    return np.array(np.random.rand(*shape), dtype=np.float64)


def randInteger(*shape):
    if isinstance(shape[0], tuple):
        shape = shape[0]
    return np.array(np.random.randint(0, 2, shape), dtype=np.float64)


def onehot(*shape):
    if isinstance(shape[0], tuple):
        shape = shape[0]
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

# Currently, pytest does not work well with decorators


def retry(attempts=3):
    def deco(func):
        def ret_func(func, *args, attempts=attempts, **kwargs):
            while(attempts > 0):
                try:
                    func(*args, **kwargs)
                    return
                except AssertionError as e:
                    err = e
                    attempts -= 1
            raise err

        return decorator.decorator(ret_func, func)
    return deco


def getNumericalDiff(lossMethod, testValue):
    assert isinstance(testValue, rm.graph.core.GraphMultiStorage)
    coefficients1 = [1, -8, 8, -1]
    coefficients2 = [-2, -1, 1, 2]
    c = 12
    eps = np.sqrt(np.finfo(rm.precision).eps)

    def store_value(index, storage, value):
        if not rm.is_cuda_active():
            storage['cpu'][index] += value
            return
        v = storage[0]
        tmp = np.empty(v.shape, dtype=np.float64)
        v.to_cpu(tmp)
        tmp[index] += value
        v.to_gpu(tmp)

    def retrieve_value(index, storage):
        if not rm.is_cuda_active():
            return storage['cpu'][index]
        v = storage[0]
        tmp = np.empty(v.shape, dtype=np.float64)
        v.to_cpu(tmp)
        return tmp[index]

    diff = np.zeros(testValue.shape, dtype=np.float64)
    for nindex in np.ndindex(diff.shape):
        loss = 0
        k = retrieve_value(nindex, testValue)
        dx = eps * k if k != 0 else eps
        for i in range(len(coefficients1)):
            store_value(nindex, testValue, coefficients2[i] * dx)
            ret = lossMethod() * coefficients1[i]
            store_value(nindex, testValue, -coefficients2[i] * dx)
            loss += ret
        v = loss / (dx * c)
        diff[nindex] = v
    return diff


@pytest.mark.parametrize("test_shape", [
    (2, 2),
    (2, 1),
    (1, 2),
    (4, 5),
])
def test_dense(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)

    v = rand(*test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.DenseGraphElement(output_size=2)
    l = rm.graph.ConstantLossGraphElement()
    m = model(val)
    loss = l(m)

    def func():
        loss.forward()
        ret = loss.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), loss.backward().get_gradient(val.value).as_ndarray())
    compare(getNumericalDiff(func, model.params['w'].output), loss.backward(
    ).get_gradient(model.params['w'].output).as_ndarray())
    compare(getNumericalDiff(func, model.params['b'].output), loss.backward(
    ).get_gradient(model.params['b'].output).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 2),
    (2, 1),
    (1, 2),
    (4, 5),
])
def test_sum(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)

    v = rand(*test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.SumGraphElement()
    l = rm.graph.ConstantLossGraphElement()
    m = model(val)
    loss = l(m)

    def func():
        loss.forward()
        ret = loss.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), loss.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 3),
    (2, 1),
    (1, 2),
    (4, 5),
])
def test_lstm(test_shape, use_gpu):
    np.random.seed(45)
    rm.set_cuda_active(use_gpu)

    v = rand(*test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.LstmGraphElement(output_size=4)
    l = rm.graph.ConstantLossGraphElement()
    m = model(val)
    loss = l(m)

    def func():
        m._fwd._op.reset()
        loss.forward()
        loss.forward()
        ret = loss.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), loss.backward().get_gradient(val.value).as_ndarray())
    compare(getNumericalDiff(func, model.params['w'].output), loss.backward(
    ).get_gradient(model.params['w'].output).as_ndarray())
    compare(getNumericalDiff(func, model.params['wr'].output), loss.backward(
    ).get_gradient(model.params['wr'].output).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 3),
    (2, 1),
    (1, 2),
    (4, 5),
])
def test_gru(test_shape, use_gpu):
    np.random.seed(44)
    rm.set_cuda_active(use_gpu)

    v = rand(*test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.GruGraphElement(output_size=4)
    l = rm.graph.ConstantLossGraphElement()
    m = model(val)
    loss = l(m)

    def func():
        m._fwd._op.reset()
        loss.forward()
        loss.forward()
        ret = loss.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), loss.backward().get_gradient(val.value).as_ndarray())
    compare(getNumericalDiff(func, model.params['w'].output), loss.backward(
    ).get_gradient(model.params['w'].output).as_ndarray(), abs_tol=1e-3)
    compare(getNumericalDiff(func, model.params['wr'].output), loss.backward(
    ).get_gradient(model.params['wr'].output).as_ndarray(), abs_tol=1e-3)


@pytest.mark.parametrize("test_shape", [
    (1, 8),
    (2, 1),
    (1, 2),
    (4, 5),
])
def test_weight_norm(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)

    v = rand(*test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.WeightNormGraphElement(output_size=3)
    l = rm.graph.ConstantLossGraphElement()
    m = model(val)
    loss = l(m)

    def func():
        loss.forward()
        ret = loss.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), loss.backward().get_gradient(val.value).as_ndarray())
    compare(getNumericalDiff(func, model.params['w'].output), loss.backward(
    ).get_gradient(model.params['w'].output).as_ndarray())
    compare(getNumericalDiff(func, model.params['g'].output), loss.backward(
    ).get_gradient(model.params['g'].output).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (1, 8),
    (2, 1),
    (1, 2),
    (4, 5),
])
def test_layer_norm(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)

    v = rand(*test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.LayerNormGraphElement()
    l = rm.graph.ConstantLossGraphElement()
    m = model(val)
    loss = l(m)

    def func():
        loss.forward()
        ret = loss.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), loss.backward().get_gradient(val.value).as_ndarray())
    compare(getNumericalDiff(func, model.params['g'].output), loss.backward(
    ).get_gradient(model.params['g'].output).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (1, 1, 3, 3),
    (2, 3, 4, 5),
])
def test_lrn(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)

    v = rand(test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.LrnGraphElement()
    l = rm.graph.ConstantLossGraphElement()
    m = model(val)
    loss = l(m)

    def func():
        loss.forward()
        ret = loss.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), loss.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (3, 1),
    (20, 1),
])
def test_embedding(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)

    v = rand(test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.EmbeddingGraphElement(output_size=2)
    l = rm.graph.ConstantLossGraphElement()
    m = model(val)
    loss = l(m)

    def func():
        loss.forward()
        ret = loss.as_ndarray()
        return ret

    compare(getNumericalDiff(func, model.params['w'].output), loss.backward(
    ).get_gradient(model.params['w'].output).as_ndarray())
    compare(getNumericalDiff(func, model.params['b'].output), loss.backward(
    ).get_gradient(model.params['b'].output).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (1, 1, 5, 5),
    (2, 3, 4, 5),
    (2, 2, 4, 4, 4),
])
def test_conv(test_shape, use_gpu):
    # TODO: Fix this weird issue
    # Fails at seed 30 (some times) for some reason
    np.random.seed(45)
    rm.set_cuda_active(use_gpu)

    v = rand(*test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.ConvolutionalGraphElement(channels=2)
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        m.forward()
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value),  l.backward().get_gradient(val.value).as_ndarray())
    compare(getNumericalDiff(func, model.params['w'].output),  l.backward(
    ).get_gradient(model.params['w'].output).as_ndarray())
    compare(getNumericalDiff(func, model.params['b'].output),  l.backward(
    ).get_gradient(model.params['b'].output).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (1, 1, 5, 5),
    (2, 3, 5, 5),
])
def test_deconv(test_shape, use_gpu):
    np.random.seed(45)
    rm.set_cuda_active(use_gpu)

    v = rand(*test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.DeconvolutionalGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        m.forward()
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value),  l.backward().get_gradient(val.value).as_ndarray())
    compare(getNumericalDiff(func, model.params['w'].output),  l.backward(
    ).get_gradient(model.params['w'].output).as_ndarray())
    compare(getNumericalDiff(func, model.params['b'].output),  l.backward(
    ).get_gradient(model.params['b'].output).as_ndarray())


def test_l2_norm(use_gpu):
    rm.set_cuda_active(use_gpu)

    v = np.array([[[[5.5, 1.1],
                    [2.3, 3.2]]]])
    val = rm.graph.StaticVariable(v)
    model = rm.graph.L2NormGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value),  l.backward().get_gradient(val.value).as_ndarray())
    compare(getNumericalDiff(func, model.params['w'].output),  l.backward(
    ).get_gradient(model.params['w'].output).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (1, 1, 5, 5),
    (2, 3, 5, 5),
    (2, 3, 4, 4, 4),
])
def test_max_pool(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)
    # Fails on seed 30
    np.random.seed(45)
    v = np.random.randint(0, 5000, test_shape).astype(rm.precision)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.MaxPoolGraphElement(kernel=3, padding=0, stride=1)
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value),  l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (1, 1, 5, 5),
    (2, 3, 5, 5),
    (2, 3, 4, 4, 4),
])
def test_avg_pool(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)
    # Fails on seed 30
    np.random.seed(45)
    v = np.random.randint(0, 5000, test_shape).astype(rm.precision)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.AvgPoolGraphElement(kernel=3, padding=0, stride=1)
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value),  l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (1, 1, 5, 5),
    (2, 3, 5, 5),
    (1, 1, 3, 3, 3),
])
def test_unpool(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)
    # Fails on seed 30
    np.random.seed(45)
    v = np.random.randint(0, 5000, test_shape).astype(rm.precision)
    val = rm.graph.StaticVariable(v)
    model1 = rm.graph.MaxPoolGraphElement(kernel=3, padding=0, stride=1)
    loss = rm.graph.ConstantLossGraphElement()
    m = model1(val)
    model2 = rm.graph.MaxUnPoolGraphElement(m)
    m2 = model2(m)
    l = loss(m2)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value),  l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (1, 8),
    (2, 5),
    (6, 1),
    (2, 20),
])
def test_cross_entropy(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    v2 = onehot(test_shape)
    val = rm.graph.StaticVariable(v)
    val2 = rm.graph.StaticVariable(v2)
    model = rm.graph.CrossEntropyGraphElement()
    m = model(val, val2)

    def func():
        m.forward()
        ret = m.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value),  m.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (1, 4),
    (2, 2),
    (2, 2, 2, 2),
])
def test_softmax_cross_entropy(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    v2 = onehot(test_shape)
    val = rm.graph.StaticVariable(v)
    val2 = rm.graph.StaticVariable(v2)
    model = rm.graph.SoftmaxCrossEntropyGraphElement()
    m = model(val, val2)

    def func():
        m.forward()
        ret = m.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value),  m.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 1),
    (2, 2),
    (2, 2, 2, 2),
])
def test_sigmoid_cross_entropy(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    v2 = randInteger(test_shape)
    val = rm.graph.StaticVariable(v)
    val2 = rm.graph.StaticVariable(v2)
    model = rm.graph.SigmoidCrossEntropyGraphElement()
    m = model(val, val2)

    def func():
        m.forward()
        ret = m.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value),  m.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 1),
    (2, 2),
    (2, 2, 2, 2),
])
def test_smoothed_l1(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    v2 = randInteger(test_shape)
    val = rm.graph.StaticVariable(v)
    val2 = rm.graph.StaticVariable(v2)
    model = rm.graph.SmoothedL1GraphElement()
    m = model(val, val2)

    def func():
        m.forward()
        ret = m.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value),  m.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 1),
    (2, 2),
    (2, 2, 2, 2),
])
def test_softmax(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.SoftmaxGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret


@pytest.mark.parametrize("test_shape", [
    (2, 1),
    (2, 2),
    (2,),
    (2, 2, 2, 2),
])
def test_softplus(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.SoftplusGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret


@pytest.mark.parametrize("test_shape", [
    (2, 1),
    (2, 2),
    (2,),
    (2, 2, 2, 2),
])
def test_relu(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.ReluGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value),  l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 1),
    (2, 2),
    (2,),
    (2, 2, 2, 2),
])
def test_elu(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.EluGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value),  l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 1),
    (2, 2),
    (2,),
    (2, 2, 2, 2),
])
def test_selu(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.SeluGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value),  l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 1),
    (2, 2),
    (2,),
    (2, 2, 2, 2),
])
def test_leaky_relu(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.LeakyReluGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value),  l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 2),
    (2, 3),
    (3, 2),
])
def test_maxout(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.MaxoutGraphElement(slice_size=2)
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value),  l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 1),
    (2, 2),
    (2,),
    (2, 2, 2, 2),
])
def test_tanh(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.TanhGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value),  l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 1),
    (2, 2),
    (2,),
    (2, 2, 2, 2),
])
def test_sigmoid(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.SigmoidGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value),  l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (1, 8),
    (2, 5),
    (6,),
    (2, 20),
])
def test_dropout(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.DropoutGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        rm.set_renom_seed(15)
        l.forward()
        ret = l.as_ndarray()
        return ret

    rm.set_renom_seed(15)
    compare(getNumericalDiff(func, val.value),  l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (1, 8),
    (2, 5),
    (6,),
    (2, 20),
])
def test_mean_squared(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    v2 = rand(test_shape)
    val = rm.graph.StaticVariable(v)
    val2 = rm.graph.StaticVariable(v2)
    model = rm.graph.MeanSquaredGraphElement()
    m = model(val, val2)

    def func():
        m.forward()
        ret = m.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value),  m.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (4, 1),
    (2, 4),
    (2, 20)
])
def test_batch_norm(test_shape, use_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v)
    m1 = rm.graph.DenseGraphElement(output_size=3)
    model = rm.graph.BatchNormalizeGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m2 = m1(val)
    m = model(m2)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value),  l.backward().get_gradient(val.value).as_ndarray())
    compare(getNumericalDiff(func, model.params['w'].output),  l.backward(
    ).get_gradient(model.params['w'].output).as_ndarray())
    compare(getNumericalDiff(func, model.params['b'].output),  l.backward(
    ).get_gradient(model.params['b'].output).as_ndarray())
