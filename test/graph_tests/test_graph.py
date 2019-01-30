import decorator
import numpy as np
import renom as rm
import pytest
from itertools import product

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


def fixed(*shape):
    if isinstance(shape[0], tuple):
        shape = shape[0]
    return np.arange(np.prod(shape)).astype(np.float64).reshape(shape)


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

        for v in storage:
            tmp = np.empty(v.shape, dtype=np.float64)
            v.to_cpu(tmp)
            tmp[index] += value
            v.to_gpu(tmp)

    def retrieve_value(index, storage):
        if not rm.is_cuda_active():
            return storage['cpu'][index]

        ret = []
        for v in storage:
            tmp = np.empty(v.shape, dtype=np.float64)
            v.to_cpu(tmp)
            ret.append(tmp[index])
        return np.mean(ret)

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


@pytest.mark.parametrize("test_shape1, test_shape2, oper", [
    *product(
        [
            (2, 2),
            (2, 1),
            (1, 1),
            (1, 2),
            (1,),
        ],
        [
            (2, 2),
            (2, 1),
            (1, 1),
            (1, 2),
            (1,),
        ],
        [
            rm.graph.core.UserGraph.__add__,
            rm.graph.core.UserGraph.__iadd__,
            rm.graph.core.UserGraph.__radd__,

            rm.graph.core.UserGraph.__sub__,
            rm.graph.core.UserGraph.__isub__,
            rm.graph.core.UserGraph.__rsub__,

            rm.graph.core.UserGraph.__mul__,
            rm.graph.core.UserGraph.__imul__,
            rm.graph.core.UserGraph.__rmul__,

            rm.graph.core.UserGraph.__div__,
            rm.graph.core.UserGraph.__idiv__,
            rm.graph.core.UserGraph.__rdiv__,

            rm.graph.core.UserGraph.__truediv__,
            rm.graph.core.UserGraph.__itruediv__,
            rm.graph.core.UserGraph.__rtruediv__,

            lambda a, b: rm.graph.AddGraphElement()(a, b),
            lambda a, b: rm.graph.DivGraphElement()(a, b),
            lambda a, b: rm.graph.MulGraphElement()(a, b),
            lambda a, b: rm.graph.SubGraphElement()(a, b),
        ]
    ),
])
def test_basic_binary_operations(test_shape1, test_shape2, oper, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    v1 = rand(*test_shape1)
    val1 = rm.graph.StaticVariable(v1, num_gpus=num_gpu)
    v2 = rand(*test_shape2)
    val2 = rm.graph.StaticVariable(v2, num_gpus=num_gpu)
    lf = rm.graph.ConstantLossGraphElement()
    loss = lf(oper(val1, val2))

    def func():
        loss.forward()
        ret = loss.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val1.value),
            loss.backward().get_gradient(val1.value).as_ndarray())
    compare(getNumericalDiff(func, val2.value),
            loss.backward().get_gradient(val2.value).as_ndarray())


@pytest.mark.parametrize("test_shape1, oper", [
    *product(
        [
            (2, 2),
            (2, 1),
            (1, 1),
            (1, 2),
            (1,),
        ],
        [
            rm.graph.basics.sqrt,
            rm.graph.basics.log,
            rm.graph.basics.square,
            rm.graph.basics.exp,
        ]
    ),
])
def test_basic_unary_operations(test_shape1, oper, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    v1 = rand(*test_shape1)
    val1 = rm.graph.StaticVariable(v1, num_gpus=num_gpu)
    lf = rm.graph.ConstantLossGraphElement()
    loss = lf(oper(val1))

    def func():
        loss.forward()
        ret = loss.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val1.value),
            loss.backward().get_gradient(val1.value).as_ndarray())


@pytest.mark.parametrize("params, oper", [
    *product(
        [
            [(2, 2), None, False],
            [(2, 2), None, True],
            [(2, 2), 0, False],
            [(2, 2), 0, True],
            [(2, 2), 1, False],
            [(2, 2), 1, True],
            [(2, 2), (0, 1), False],
            [(2, 2), (0, 1), True],
            [(2, 2), (1, 0), False],
            [(2, 2), (1, 0), True],
            [(2, 2, 2), None, False],
            [(2, 2, 2), None, True],
            [(2, 2, 2), 0, False],
            [(2, 2, 2), 0, True],
            [(2, 2, 2), 1, False],
            [(2, 2, 2), 1, True],
            [(2, 2, 2), 2, False],
            [(2, 2, 2), 2, True],
            [(2, 2, 2), (0, 1), False],
            [(2, 2, 2), (0, 1), True],
            [(2, 2, 2), (1, 0), False],
            [(2, 2, 2), (1, 0), True],
            [(2, 2, 2), (0, 2), False],
            [(2, 2, 2), (0, 2), True],
            [(2, 2, 2), (1, 2), False],
            [(2, 2, 2), (1, 2), True],
            [(2, 2, 2), (0, 1, 2), False],
            [(2, 2, 2), (0, 1, 2), True],

            [(2, 2, 2, 2), (2, 3), False],
            [(2, 2, 2, 2), (2, 3), True],

            [(2, 2, 3, 4), (2, 3), False],
            [(2, 2, 3, 4), (2, 3), True],
        ],
        [
            rm.graph.basics.sum,
            rm.graph.basics.mean,
            lambda a, axis, keepdims: rm.graph.MeanGraphElement(axis, keepdims)(a),
            lambda a, axis, keepdims: rm.graph.SumGraphElement(axis, keepdims)(a),
        ]
    ),
])
def test_reduce_unary_operations1(params, oper, use_gpu, num_gpu):
    test_shape1, axis, keepdims = params
    rm.set_cuda_active(use_gpu)
    v1 = rand(*test_shape1)
    val1 = rm.graph.StaticVariable(v1, num_gpus=num_gpu)
    lf = rm.graph.ConstantLossGraphElement()
    loss = lf(oper(val1, axis=axis, keepdims=keepdims))

    def func():
        loss.forward()
        ret = loss.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val1.value),
            loss.backward().get_gradient(val1.value).as_ndarray())


@pytest.mark.parametrize("params, oper", [
    *product(
        [
            [(2, 2), None, False],
            [(2, 2), None, True],
            [(2, 2), 0, False],
            [(2, 2), 0, True],
            [(2, 2), 1, False],
            [(2, 2), 1, True],
            [(2, 2, 2), None, False],
            [(2, 2, 2), None, True],
            [(2, 2, 2), 0, False],
            [(2, 2, 2), 0, True],
            [(2, 2, 2), 1, False],
            [(2, 2, 2), 1, True],
            [(2, 2, 2), 2, False],
            [(2, 2, 2), 2, True],
            [(2, 2, 2, 2), 0, False],
            [(2, 2, 2, 2), 0, True],
            [(2, 2, 2, 2), 1, False],
            [(2, 2, 2, 2), 1, True],
            [(2, 2, 2, 2), 2, False],
            [(2, 2, 2, 2), 2, True],
            [(2, 2, 2, 2), 3, False],
            [(2, 2, 2, 2), 3, True],
        ],
        [
            rm.graph.basics.min,
            rm.graph.basics.max,
        ]
    ),
])
def test_reduce_unary_operations2(params, oper, use_gpu, num_gpu):
    test_shape1, axis, keepdims = params
    rm.set_cuda_active(use_gpu)
    v1 = fixed(*test_shape1)
    val1 = rm.graph.StaticVariable(v1, num_gpus=num_gpu)
    lf = rm.graph.ConstantLossGraphElement()
    loss = lf(oper(val1, axis=axis, keepdims=keepdims))

    def func():
        loss.forward()
        ret = loss.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val1.value),
            loss.backward().get_gradient(val1.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 2),
    (2, 1),
    (1, 2),
    (4, 5),
])
def test_dense(test_shape, use_gpu, num_gpu, ignore_bias):
    rm.set_cuda_active(use_gpu)
    v = rand(*test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    model = rm.graph.Dense(output_size=2, ignore_bias=ignore_bias)
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
    if not ignore_bias:
        compare(getNumericalDiff(func, model.params['b'].output), loss.backward(
        ).get_gradient(model.params['b'].output).as_ndarray())


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
    model = rm.graph.Lstm(output_size=4)
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
def test_gru(test_shape, use_gpu, ignore_bias):
    np.random.seed(44)
    rm.set_cuda_active(use_gpu)

    v = rand(*test_shape)
    val = rm.graph.StaticVariable(v)
    model = rm.graph.GruGraphElement(output_size=4, ignore_bias=ignore_bias)
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
def test_weight_norm(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)

    v = rand(*test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
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
def test_layer_norm(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)

    v = rand(*test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
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
def test_lrn(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)

    v = rand(test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
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
def test_embedding(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)

    v = rand(test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
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
def test_conv(test_shape, use_gpu, num_gpu, ignore_bias):
    # TODO: Fix this weird issue
    # Fails at seed 30 (some times) for some reason
    np.random.seed(45)
    rm.set_cuda_active(use_gpu)

    v = rand(*test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    model = rm.graph.ConvolutionalGraphElement(channels=2, ignore_bias=ignore_bias)
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        m.forward()
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), l.backward().get_gradient(val.value).as_ndarray())
    compare(getNumericalDiff(func, model.params['w'].output), l.backward(
    ).get_gradient(model.params['w'].output).as_ndarray())
    compare(getNumericalDiff(func, model.params['b'].output), l.backward(
    ).get_gradient(model.params['b'].output).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (1, 1, 5, 5),
    (2, 3, 5, 5),
])
def test_deconv(test_shape, use_gpu, num_gpu):
    np.random.seed(45)
    rm.set_cuda_active(use_gpu)

    v = rand(*test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    model = rm.graph.DeconvolutionalGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        m.forward()
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), l.backward().get_gradient(val.value).as_ndarray())
    compare(getNumericalDiff(func, model.params['w'].output), l.backward(
    ).get_gradient(model.params['w'].output).as_ndarray())
    compare(getNumericalDiff(func, model.params['b'].output), l.backward(
    ).get_gradient(model.params['b'].output).as_ndarray())


def test_l2_norm(use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)

    v = np.array([[[[5.5, 1.1],
                    [2.3, 3.2]]]])
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    model = rm.graph.L2NormGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), l.backward().get_gradient(val.value).as_ndarray())
    compare(getNumericalDiff(func, model.params['w'].output), l.backward(
    ).get_gradient(model.params['w'].output).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (1, 1, 5, 5),
    (2, 3, 5, 5),
    (2, 3, 4, 4, 4),
])
def test_max_pool(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    # Fails on seed 30
    np.random.seed(45)
    v = np.random.randint(0, 5000, test_shape).astype(rm.precision)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    model = rm.graph.MaxPoolGraphElement(kernel=3, padding=0, stride=1)
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (1, 1, 5, 5),
    (2, 3, 5, 5),
    (2, 3, 4, 4, 4),
])
def test_avg_pool(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    # Fails on seed 30
    np.random.seed(45)
    v = np.random.randint(0, 5000, test_shape).astype(rm.precision)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    model = rm.graph.AvgPoolGraphElement(kernel=3, padding=0, stride=1)
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (1, 1, 5, 5),
    (2, 3, 5, 5),
    (1, 1, 3, 3, 3),
])
def test_unpool(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    # Fails on seed 30
    np.random.seed(45)
    v = np.random.randint(0, 5000, test_shape).astype(rm.precision)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
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

    compare(getNumericalDiff(func, val.value), l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (1, 8),
    (2, 5),
    (6, 1),
    (2, 20),
])
def test_cross_entropy(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    v2 = onehot(test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    val2 = rm.graph.StaticVariable(v2, num_gpus=num_gpu)
    model = rm.graph.CrossEntropyGraphElement()
    m = model(val, val2)

    def func():
        m.forward()
        ret = m.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), m.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (1, 4),
    (2, 2),
    (2, 2, 2, 2),
])
def test_softmax_cross_entropy(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    v2 = onehot(test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    val2 = rm.graph.StaticVariable(v2, num_gpus=num_gpu)
    model = rm.graph.SoftmaxCrossEntropyGraphElement()
    m = model(val, val2)

    def func():
        m.forward()
        ret = m.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), m.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 1),
    (2, 2),
    (2, 2, 2, 2),
])
def test_sigmoid_cross_entropy(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    v2 = randInteger(test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    val2 = rm.graph.StaticVariable(v2, num_gpus=num_gpu)
    model = rm.graph.SigmoidCrossEntropyGraphElement()
    m = model(val, val2)

    def func():
        m.forward()
        ret = m.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), m.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 1),
    (2, 2),
    (2, 2, 2, 2),
])
def test_smoothed_l1(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    v2 = randInteger(test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    val2 = rm.graph.StaticVariable(v2, num_gpus=num_gpu)
    model = rm.graph.SmoothedL1GraphElement()
    m = model(val, val2)

    def func():
        m.forward()
        ret = m.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), m.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 1),
    (2, 2),
    (2, 2, 2, 2),
])
def test_softmax(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
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
def test_softplus(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
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
def test_relu(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    model = rm.graph.ReluGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 1),
    (2, 2),
    (2,),
    (2, 2, 2, 2),
])
def test_elu(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    model = rm.graph.EluGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 1),
    (2, 2),
    (2,),
    (2, 2, 2, 2),
])
def test_selu(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    model = rm.graph.SeluGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 1),
    (2, 2),
    (2,),
    (2, 2, 2, 2),
])
def test_leaky_relu(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    model = rm.graph.LeakyReluGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 2),
    (2, 3),
    (3, 2),
])
def test_maxout(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    model = rm.graph.MaxoutGraphElement(slice_size=2)
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 1),
    (2, 2),
    (2,),
    (2, 2, 2, 2),
])
def test_tanh(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    model = rm.graph.TanhGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (2, 1),
    (2, 2),
    (2,),
    (2, 2, 2, 2),
])
def test_sigmoid(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    model = rm.graph.SigmoidGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (1, 8),
    (2, 5),
    (6,),
    (2, 20),
])
def test_dropout(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    model = rm.graph.DropoutGraphElement()
    loss = rm.graph.ConstantLossGraphElement()
    m = model(val)
    l = loss(m)

    def func():
        rm.set_renom_seed(15, all_devices=True)
        l.forward()
        ret = l.as_ndarray()
        return ret

    rm.set_renom_seed(15, all_devices=True)
    compare(getNumericalDiff(func, val.value), l.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (1, 8),
    (2, 5),
    (6,),
    (2, 20),
])
def test_mean_squared(test_shape, use_gpu, num_gpu):
    rm.set_cuda_active(use_gpu)
    v = rand(test_shape)
    v2 = rand(test_shape)
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    val2 = rm.graph.StaticVariable(v2, num_gpus=num_gpu)
    model = rm.graph.MeanSquaredGraphElement()
    m = model(val, val2)

    def func():
        m.forward()
        ret = m.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), m.backward().get_gradient(val.value).as_ndarray())


@pytest.mark.parametrize("test_shape", [
    (4, 4),
    (2, 4),
    (2, 20),
    (3, 2, 4, 5),
])
def test_batch_norm(test_shape, use_gpu, num_gpu, ignore_bias):
    rm.set_cuda_active(use_gpu)
    rm.set_renom_seed(45)
    v = rand(test_shape)
    if len(test_shape) > 2:
        mode = rm.graph.batch_normalize_element.BATCH_NORMALIZE_FEATUREMAP
    else:
        mode = rm.graph.batch_normalize_element.BATCH_NORMALIZE_ELEMENTWISE
    val = rm.graph.StaticVariable(v, num_gpus=num_gpu)
    model = rm.graph.BatchNormalizeGraphElement(mode=mode, ignore_bias=ignore_bias)
    loss = rm.graph.ConstantLossGraphElement()
    m2 = model(val)
    m = model(m2)
    l = loss(m)

    def func():
        l.forward()
        ret = l.as_ndarray()
        return ret

    compare(getNumericalDiff(func, val.value), l.backward().get_gradient(val.value).as_ndarray())
    compare(getNumericalDiff(func, model.params['w'].output), l.backward(
    ).get_gradient(model.params['w'].output).as_ndarray())
    compare(getNumericalDiff(func, model.params['b'].output), l.backward(
    ).get_gradient(model.params['b'].output).as_ndarray())
