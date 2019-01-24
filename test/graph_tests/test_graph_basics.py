import renom as rm
import numpy as np
import pytest


def compare(nd_value, ad_value):
    print('ad=')
    print(ad_value)
    print('nd=')
    print(nd_value)
    assert np.allclose(nd_value, ad_value, atol=1e-5, rtol=1e-3)


def get_random_filename():
    import random
    import string
    pre_filename = 'tmpfile-'
    rand_filename = ''.join(random.choice(string.ascii_uppercase + string.digits)
                            for _ in range(11))
    type_filename = '.h5'
    tmp_filename = pre_filename + rand_filename + type_filename
    return tmp_filename


def test_basic_add():

    v1 = np.random.rand(2, 2)
    v2 = np.random.rand(2, 2)
    v3 = v1 + v2
    v4 = np.random.rand(2, 2)
    v5 = v3 + v4

    g1 = rm.graph.StaticVariable(v1)
    g2 = rm.graph.StaticVariable(v2)
    g3 = g1 + g2
    g4 = rm.graph.StaticVariable(v4)
    g5 = g3 + g4

    compare(v5, g5.as_ndarray())

    new_v1 = np.random.rand(2, 2)
    g1.value = new_v1

    new_v5 = new_v1 + v2 + v4
    g5.forward()
    compare(new_v5, g5.as_ndarray())


def test_basic_lstm():

    np.random.seed(45)
    v = np.random.rand(2, 2)
    layer = rm.graph.LstmGraphElement(3)
    t = np.random.rand(2, 3)
    loss = rm.graph.MeanSquaredGraphElement()
    opt = rm.graph.sgd_update(0.01, 0.4)
    p_l = 9999
    for i in range(3):
        layer.reset()
        l = loss(layer(v), t)
        l_arr = l.as_ndarray()
        assert l_arr < p_l
        p_l = l_arr
        l.backward().update(opt)


def test_slices(use_gpu):
    rm.set_cuda_active(use_gpu)

    a = np.random.rand(3, 3, 3)
    A = rm.graph.StaticVariable(a)
    b = a[:, 1, 0:2]
    B = A[:, 1, 0:2]
    compare(b, B.as_ndarray())


def test_optimizer(use_gpu):

    rm.set_cuda_active(use_gpu)
    np.random.seed(45)
    v = np.random.rand(2, 2)
    layer = rm.graph.DenseGraphElement(3)
    t = np.random.rand(2, 3)
    loss = rm.graph.MeanSquaredGraphElement()
    opt = rm.graph.adam_update()
    p_l = 9999
    for i in range(5):
        l = loss(layer(v), t)
        l_arr = l.as_ndarray()
        assert l_arr < p_l
        p_l = l_arr
        l.backward().update(opt)


@pytest.mark.skipif(rm.precision != np.float64, reason='Requires precise testing')
def test_inference_executor(use_gpu):
    rm.set_cuda_active(use_gpu)

    np.random.seed(45)
    v = np.random.rand(20, 3).astype(rm.precision)
    layer = rm.graph.DenseGraphElement(4)
    t = np.random.rand(20, 4).astype(rm.precision)
    loss = rm.graph.MeanSquaredGraphElement()
    data, target = rm.graph.DistributorElement(v, t, batch_size=2).getOutputGraphs()
    exe = loss(layer(data), target).getInferenceExecutor()
    losses = exe.execute(epochs=3)
    assert all(losses[i] == losses[i + 1] for i in range(len(losses) - 1))


def test_training_executor(use_gpu):
    rm.set_cuda_active(use_gpu)

    np.random.seed(45)
    v = np.random.rand(20, 3).astype(rm.precision)
    layer = rm.graph.DenseGraphElement(4)
    t = np.random.rand(20, 4).astype(rm.precision)
    loss = rm.graph.MeanSquaredGraphElement()
    opt = rm.graph.sgd_update()
    data, target = rm.graph.DistributorElement(v, t, batch_size=2).getOutputGraphs()
    exe = loss(layer(data), target).getTrainingExecutor(opt)
    losses = exe.execute(epochs=3)
    assert all(losses[i] >= losses[i + 1] for i in range(len(losses) - 1))


def test_validation_executor(use_gpu):
    rm.set_cuda_active(use_gpu)

    np.random.seed(45)
    v1 = np.random.rand(10, 2).astype(rm.precision)
    layer = rm.graph.DenseGraphElement(4)
    t1 = np.random.rand(10, 4).astype(rm.precision)
    loss = rm.graph.MeanSquaredGraphElement()
    data, target = rm.graph.DistributorElement(v1, t1, batch_size=2).getOutputGraphs()
    exe = loss(layer(data), target).getInferenceExecutor()
    losses1 = np.array(exe.execute(epochs=3))
    v2, t2 = v1 * 2, t1 * 2
    exe.set_input_data(v2, t2)
    losses2 = np.array(exe.execute(epochs=3))
    assert np.allclose(losses1 * 4, losses2, atol=1)


def test_step_executor(use_gpu):
    rm.set_cuda_active(use_gpu)
    rm.set_cuda_active(use_gpu)

    np.random.seed(45)
    v1 = np.random.rand(10, 2).astype(rm.precision)
    layer = rm.graph.DenseGraphElement(4)
    t1 = np.random.rand(10, 4).astype(rm.precision)
    loss = rm.graph.MeanSquaredGraphElement()
    data, target = rm.graph.DistributorElement(v1, t1, batch_size=2).getOutputGraphs()
    exe = loss(layer(data), target).getInferenceExecutor()
    loss1 = np.array(exe.execute(epochs=1))
    loss2 = 0
    for i in range(0, 10, 2):
        v2, t2 = v1[i:i + 2] * 2, t1[i:i + 2] * 2
        loss2 += exe.step(v2, t2)
    assert np.allclose(loss1 * 4, loss2, atol=1)


def test_finalizer(use_gpu):
    rm.set_cuda_active(use_gpu)

    np.random.seed(45)
    v = np.random.rand(2, 1, 3, 4)
    layer1 = rm.graph.ConvolutionalGraphElement(channels=2)
    res = rm.graph.ReshapeGraphElement([-1])
    layer2 = rm.graph.DenseGraphElement(3)
    t = np.random.rand(2, 3)
    loss = rm.graph.MeanSquaredGraphElement()
    opt = rm.graph.sgd_update()

    z = v
    z = layer1(z)
    z = res(z)
    z = layer2(z)
    z = loss(z, t)
    z._fwd.finalize()


def test_sequential(use_gpu):
    rm.set_cuda_active(use_gpu)

    np.random.seed(45)
    v = np.random.rand(4, 4)
    model = rm.graph.SequentialSubGraph([
        rm.graph.DenseGraphElement(3),
        rm.graph.DenseGraphElement(1),
        rm.graph.DenseGraphElement(5),
    ])
    z = model(v).as_ndarray()
    assert z.shape == (4, 5)


def test_weight_decay(use_gpu):
    rm.set_cuda_active(use_gpu)

    np.random.seed(45)
    v = np.random.rand(4, 4)
    dense = rm.graph.DenseGraphElement(3, weight_decay=0.05)
    import os
    tmp_filename = get_random_filename()
    try:
        m1 = dense(v)
        m_arr1 = m1.as_ndarray()
        dense.save(tmp_filename)
        m1.backward().update()
        w1 = dense.params['w'].as_ndarray()

        dense.load(tmp_filename)
        dense.params['w'].set_weight_decay(0.50)
        m2 = dense(v)
        m_arr2 = m2.as_ndarray()
        m2.backward().update()
        w2 = dense.params['w'].as_ndarray()
        assert np.allclose(m_arr1, m_arr2)
        assert not np.allclose(w1, w2)
    except Exception as e:
        os.remove(tmp_filename)
        raise e
    os.remove(tmp_filename)


class noop(rm.graph.core.operation):
    name = 'noop'
    _vars = {'y': rm.graph.core.GraphMultiStorage(shape=(0,), gpus='cpu')}

    def setup(self, inputs):
        pass

    def perform(self):
        pass


@pytest.mark.parametrize('graph_nodes', [
    {'A': rm.graph.core.operational_element(operation=noop(), tags=['Dummy']),
        'B': rm.graph.core.operational_element(operation=noop(), tags=['Dummy']),
        'C': rm.graph.core.operational_element(operation=noop(), tags=['Dummy'])
     },
    {'A': rm.graph.core.UserGraph(forward_operation=noop()),
        'B': rm.graph.core.UserGraph(forward_operation=noop()),
        'C': rm.graph.core.UserGraph(forward_operation=noop())
     },
])
def test_graph_depth(graph_nodes):
    A = graph_nodes['A']
    B = graph_nodes['B']
    C = graph_nodes['C']
    assert A.depth == 0 and B.depth == 0 and C.depth == 0

    B.add_input(A)  # A(0) -> B(1), C(0)
    assert A.depth == 0 and B.depth == 1 and C.depth == 0

    C.add_input(B)  # A(0) -> B(1) -> C(2)
    assert A.depth == 0 and B.depth == 1 and C.depth == 2

    C.detach()      # A(0) -> B(1), C(0)
    assert len(C._next_elements) == 0 and len(C._previous_elements) == 0
    assert A.depth == 0 and B.depth == 1 and C.depth == 0

    C.add_input(B)  # A(0) -> B(1) -> C(2)
    assert A.depth == 0 and B.depth == 1 and C.depth == 2

    B.detach()      # A(0), B(0), C(0)
    assert len(B._next_elements) == 0 and len(B._previous_elements) == 0
    assert A.depth == 0 and B.depth == 0 and C.depth == 0
    assert len(B._previous_elements) == 0 and len(B._next_elements) == 0

    C.add_input(A)  # A(0) -> C(1), B(0)
    assert A.depth == 0 and B.depth == 0 and C.depth == 1

    B.add_input(A)  # A(0) -> (B(1) , C(1))
    assert A.depth == 0 and B.depth == 1 and C.depth == 1

    B.add_input(A)  # A(0) -> (B(1) , C(1))
    assert A.depth == 0 and B.depth == 1 and C.depth == 1

    B.detach()      # A(0) -> C(1), B(0)
    assert len(B._next_elements) == 0 and len(B._previous_elements) == 0
    assert A.depth == 0 and B.depth == 0 and C.depth == 1


@pytest.mark.parametrize('A_has_back', [True, False])
@pytest.mark.parametrize('B_has_back', [True, False])
@pytest.mark.parametrize('C_has_back', [True, False])
def test_user_graph_connection(A_has_back, B_has_back, C_has_back):
    rm.set_cuda_active(False)

    A = rm.graph.core.UserGraph(forward_operation=noop(), backward_operations=[
        noop()] if A_has_back else None)
    B = rm.graph.core.UserGraph(forward_operation=noop(), backward_operations=[
        noop()] if B_has_back else None)
    C = rm.graph.core.UserGraph(forward_operation=noop(), backward_operations=[
        noop()] if C_has_back else None)
    assert A.depth == 0 and B.depth == 0 and C.depth == 0
    assert A_has_back and len(A._bwd_graphs) == 1 or len(A._bwd_graphs) == 0

    # A_f -> B_f, B_b -> A_b, C_f, C_b
    B(A)
    assert A.depth == 0 and B.depth == 1 and C.depth == 0
    assert A._fwd.depth == 0 and B._fwd.depth == 1 and C._fwd.depth == 0
    if A_has_back and B_has_back:
        assert A._bwd_graphs[0].depth == 1 and B._bwd_graphs[0].depth == 0

    # A_f, A_b, B_f, B_b, C_f, C_b
    B.detach()
    assert len(B._next_elements) == 0 and len(B._previous_elements) == 0
    assert A.depth == 0 and B.depth == 0 and C.depth == 0
    assert A._fwd.depth == 0 and B._fwd.depth == 0 and C._fwd.depth == 0
    if A_has_back and B_has_back:
        assert A._bwd_graphs[0].depth == 0 and B._bwd_graphs[0].depth == 0

    # A_f -> B_f, B_b -> A_b, C_f, C_b
    B(A)
    assert A.depth == 0 and B.depth == 1 and C.depth == 0
    assert A._fwd.depth == 0 and B._fwd.depth == 1 and C._fwd.depth == 0
    if A_has_back and B_has_back:
        assert A._bwd_graphs[0].depth == 1 and B._bwd_graphs[0].depth == 0

    # A_f -> B_f -> C_f, C_b -> B_b -> A_b
    C(B)
    assert A.depth == 0 and B.depth == 1 and C.depth == 2
    assert A._fwd.depth == 0 and B._fwd.depth == 1 and C._fwd.depth == 2
    if C_has_back and B_has_back and A_has_back:
        assert C._bwd_graphs[0].depth == 0 and B._bwd_graphs[0].depth == 1 and A._bwd_graphs[0].depth == 2
    elif C_has_back and B_has_back and not A_has_back:
        assert C._bwd_graphs[0].depth == 0 and B._bwd_graphs[0].depth == 1
    elif not C_has_back and B_has_back and A_has_back:
        assert B._bwd_graphs[0].depth == 0 and A._bwd_graphs[0].depth == 1

    # A_f, A_b, B_f, B_b, C_f, C_b
    B.detach()
    assert len(B._next_elements) == 0 and len(B._previous_elements) == 0
    assert A.depth == 0 and B.depth == 0 and C.depth == 0
    assert A._fwd.depth == 0 and B._fwd.depth == 0 and C._fwd.depth == 0
    if A_has_back and B_has_back:
        assert A._bwd_graphs[0].depth == 0 and B._bwd_graphs[0].depth == 0

    # A_f -> B_f -> C_f, C_b -> B_b -> A_b
    C(B(A))
    assert A.depth == 0 and B.depth == 1 and C.depth == 2
    assert A._fwd.depth == 0 and B._fwd.depth == 1 and C._fwd.depth == 2
    if C_has_back and B_has_back and A_has_back:
        assert C._bwd_graphs[0].depth == 0 and B._bwd_graphs[0].depth == 1 and A._bwd_graphs[0].depth == 2
    elif C_has_back and B_has_back and not A_has_back:
        assert C._bwd_graphs[0].depth == 0 and B._bwd_graphs[0].depth == 1
    elif not C_has_back and B_has_back and A_has_back:
        assert B._bwd_graphs[0].depth == 0 and A._bwd_graphs[0].depth == 1

    if A_has_back and B_has_back and C_has_back:
        L = rm.graph.core.UserLossGraph(forward_operation=noop(), backward_operations=[noop()])
        # A_f -> B_f -> C_f -> L_f -> L_b -> C_b -> B_b -> A_b
        L(C)
        assert A.depth == 0 and B.depth == 1 and C.depth == 2 and L.depth == 3
        assert A._fwd.depth == 0 and B._fwd.depth == 1 and C._fwd.depth == 2 and L._fwd.depth == 3
        assert L._bwd_graphs[0].depth == 4 and C._bwd_graphs[0].depth == 5 \
            and B._bwd_graphs[0].depth == 6 and A._bwd_graphs[0].depth == 7

        # A_f -> B_f -> C_f, C_b -> B_b -> A_b
        L.detach()
        assert A.depth == 0 and B.depth == 1 and C.depth == 2
        assert A._fwd.depth == 0 and B._fwd.depth == 1 and C._fwd.depth == 2
        if C_has_back and B_has_back and A_has_back:
            assert C._bwd_graphs[0].depth == 0 and B._bwd_graphs[0].depth == 1 and A._bwd_graphs[0].depth == 2
        elif C_has_back and B_has_back and not A_has_back:
            assert C._bwd_graphs[0].depth == 0 and B._bwd_graphs[0].depth == 1
        elif not C_has_back and B_has_back and A_has_back:
            assert B._bwd_graphs[0].depth == 0 and A._bwd_graphs[0].depth == 1

        # A_f -> B_f -> C_f -> L_f -> L_b -> C_b -> B_b -> A_b
        C.backward()
        assert A.depth == 0 and B.depth == 1 and C.depth == 2
        assert A._fwd.depth == 0 and B._fwd.depth == 1 and C._fwd.depth == 2
        assert C._bwd_graphs[0].depth == 5 and B._bwd_graphs[0].depth == 6 \
            and A._bwd_graphs[0].depth == 7


def get_random_filename():
    import random
    import string
    pre_filename = 'tmpfile-'
    rand_filename = ''.join(random.choice(string.ascii_uppercase + string.digits)
                            for _ in range(11))
    type_filename = '.h5'
    tmp_filename = pre_filename + rand_filename + type_filename
    return tmp_filename


@pytest.mark.parametrize('devices_to_load', [
    'cpu', 1, 2, 3, 4
])
def test_save_load(devices_to_load):
    if devices_to_load != 'cpu':
        if not rm.cuda.has_cuda() or (rm.cuda.cuGetDeviceCount() < devices_to_load):
            pytest.skip()
        rm.set_cuda_active(True)
        devices_to_load = [d for d in range(devices_to_load)]
    else:
        rm.set_cuda_active(False)

    model = rm.graph.SequentialSubGraph([
        rm.graph.DenseGraphElement(3),
        rm.graph.DenseGraphElement(6),
        rm.graph.DenseGraphElement(2),
    ])

    x = np.random.rand(5, 4)
    y1 = model(x).as_ndarray()

    tmp_filename = get_random_filename()
    model.save(tmp_filename)

    model = rm.graph.SequentialSubGraph([
        rm.graph.DenseGraphElement(3),
        rm.graph.DenseGraphElement(6),
        rm.graph.DenseGraphElement(2),
    ])
    model.load(tmp_filename, devices=devices_to_load)
    y2 = model(x).as_ndarray()
    assert np.allclose(y1, y2)

    try:
        model = rm.graph.SequentialSubGraph([
            rm.graph.DenseGraphElement(6),
            rm.graph.DenseGraphElement(3),
            rm.graph.DenseGraphElement(2),
        ])
        model.load(tmp_filename)
        raise AssertionError('Model should not be able to load different shape')
    except:
        pass

    import os
    os.remove(tmp_filename)


def test_version_save_compability(use_gpu):
    rm.set_cuda_active(use_gpu)

    x = np.random.rand(1, 4)

    v2_model = rm.Sequential([
        rm.Dense(5),
        rm.Dense(3),
        rm.Dense(1),
    ])

    y1 = v2_model(x)
    tmp_filename = get_random_filename()
    v2_model.save(tmp_filename)
    v3_model = rm.graph.SequentialSubGraph([
        rm.graph.DenseGraphElement(5),
        rm.graph.DenseGraphElement(3),
        rm.graph.DenseGraphElement(1),
    ])
    v3_model.load(tmp_filename)
    y2 = v3_model(x).as_ndarray()
    assert np.allclose(y1, y2)

    import os
    os.remove(tmp_filename)


@pytest.mark.parametrize('ttype', [
    np.int, np.int32, np.int64,
    np.float, np.float32, np.float64
])
def test_dtype(ttype, use_gpu):
    rm.set_cuda_active(use_gpu)

    model = rm.graph.SequentialSubGraph([
        rm.graph.DenseGraphElement(3),
        rm.graph.DenseGraphElement(6),
        rm.graph.DenseGraphElement(2),
    ])

    x = np.random.rand(5, 4).astype(ttype)
    y1 = model(x)
    assert y1.as_ndarray().dtype == rm.precision


def test_pinnedmem():  # TODO
    pytest.skip()
