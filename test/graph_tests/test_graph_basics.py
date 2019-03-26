import renom as rm
import renom.graph as rmg
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

    g1 = rmg.StaticVariable(v1)
    g2 = rmg.StaticVariable(v2)
    g3 = g1 + g2
    g4 = rmg.StaticVariable(v4)
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
    layer = rmg.Lstm(3)
    t = np.random.rand(2, 3)
    loss = rmg.MeanSquared()
    opt = rmg.Sgd(0.01, 0.4)
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
    A = rmg.StaticVariable(a)
    b = a[:, 1, 0:2]
    B = A[:, 1, 0:2]
    compare(b, B.as_ndarray())


def test_distributor_test_split(use_gpu):
    rm.set_cuda_active(use_gpu)

    a = np.random.rand(10, 2).astype(rm.precision)
    b = np.random.rand(10, 4).astype(rm.precision)
    #data, target = rm.graph.Distro(a, b, batch_size=2, test_split=0.8).get_output_graphs()
    data, target = rmg.DataInput([a[:8], b[:8]]).shuffle().batch(2).get_output_graphs()
    data.reset()
    model = rm.graph.Dense(3)
    count = 0
    try:
        while(True):
            data.forward()
            # target.forward()
            count += 1
            x = model(data)
    except StopIteration:
        pass
    assert count == 5
    data, target = rmg.DataInput([a[8:], b[8:]]).shuffle().batch(2).get_output_graphs()
    data.reset()
    count = 0
    try:
        while(True):
            data.forward()
            count += 1
            x = model(data)
    except StopIteration:
        pass
    assert count == 2


class BadSgd(rm.graph.utils.optimizer.optimizer_factory):

    class gpu_op:

        def setup(self, grad, val):
            self._dy = grad
            self._outputs = val
            self.gpus = grad.gpus

        def update(self):
            for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
                self._outputs[gpu] += self._dy[gpu]

    class cpu_op(gpu_op):

        def update(self):
            dy = self._dy['cpu']
            self._outputs['cpu'] += 0.001 * dy


def test_split_backwards(use_gpu):
    rm.set_cuda_active(use_gpu)

    np.random.seed(45)
    v1 = rmg.StaticVariable(np.random.rand(2, 2))
    v2 = rmg.StaticVariable(np.random.rand(2, 2))
    layer = rmg.Dense(3)
    act1 = rmg.Tanh()
    act2 = rmg.Sigmoid()
    l1 = act1(layer(v1))
    l2 = act2(layer(v2))
    ll = l1 + l2
    ll.backward()
    gv1 = ll.get_gradient(v1.value)
    gv2 = ll.get_gradient(v2.value)
    assert not np.allclose(gv1, gv2)


def test_no_graph_artifacts():

    v1 = rmg.StaticVariable(np.random.rand(2, 2))
    v2 = rmg.StaticVariable(np.random.rand(2, 2))
    layer = rmg.Dense(3)
    l1 = layer(v1)
    l1.backward()
    l2 = layer(v2)
    l2.backward()
    l1.get_gradient(v1.value)
    try:
        l2.get_gradient(v1.value)
        assert False, 'Graph artifacts occured in posterior graph'
    except AttributeError:
        pass

    try:
        l1.get_gradient(v2.value)
        assert False, 'Graph artifacts occured in prior graph'
    except AttributeError:
        pass


def test_diamond_shared():
    v1 = rm.graph.StaticVariable(np.random.rand(2, 3))
    t1 = rm.graph.StaticVariable(np.random.rand(2, 1))
    l1 = rmg.Dense(2)
    l21 = rmg.Dense(3)
    l22 = rmg.Dense(1)
    ls = rmg.MeanSquared()
    g1 = l1(v1)
    with l21.no_updates():
        g21 = l22(l21(g1))
    with l22.no_updates():
        g22 = l22(l21(g1))
    l = ls(g21 + g22, t1)


def test_optimizer(use_gpu):

    rm.set_cuda_active(use_gpu)
    np.random.seed(45)
    v = np.random.rand(2, 2)
    layer = rmg.Dense(3)
    t = np.random.rand(2, 3)
    loss = rmg.MeanSquared()
    opt1 = BadSgd()
    opt2 = rmg.Sgd()
    p_l = 0
    for i in range(5):
        l = loss(layer(v), t)
        l_arr = l.as_ndarray()
        assert l_arr > p_l
        p_l = l_arr
        l.backward().update(opt1)
    p_l = 9999999
    for i in range(5):
        print(i)
        l = loss(layer(v), t)
        l_arr = l.as_ndarray()
        assert l_arr < p_l
        p_l = l_arr
        l.backward().update(opt2)


def test_inference_executor(use_gpu):
    rm.set_cuda_active(use_gpu)

    np.random.seed(42)
    v = np.random.rand(20, 3).astype(rm.precision)
    layer = rmg.Dense(4)
    t = np.random.rand(20, 4).astype(rm.precision)
    loss = rmg.MeanSquared()
    data, target = rmg.DataInput([v, t]).shuffle().batch(2).get_output_graphs()
    exe = loss(layer(data), target).get_executor()
    losses = []

    def add_losses(info):
        epoch_loss_list = info['epoch_loss_list']
        losses.append(np.sum(epoch_loss_list))
    exe.register_event('Epoch-Finish', add_losses)
    exe.execute(epochs=3)
    assert all(np.allclose(losses[i], losses[i + 1]) for i in range(len(losses) - 2))


def test_training_executor(use_gpu):
    rm.set_cuda_active(use_gpu)

    np.random.seed(25)
    v = np.random.rand(500, 3).astype(rm.precision)
    layer = rmg.Dense(2)
    t = np.random.rand(500, 2).astype(rm.precision)
    loss = rmg.MeanSquared(reduction='sum')
    opt = rmg.Sgd(0.01)
    data, target = rmg.DataInput([v, t]).shuffle().batch(10).get_output_graphs()
    exe = loss(rmg.relu(layer(data)), target).get_executor(optimizer=opt, mode='training')
    losses = []

    def add_losses(info):
        epoch_loss_list = info['epoch_loss_list']
        losses.append(np.sum(epoch_loss_list))
    exe.register_event('Epoch-Finish', add_losses)
    exe.execute(epochs=3)
    print(losses)
    assert all(losses[i] >= losses[i + 1] for i in range(len(losses) - 1))


def test_training_executor_validation(use_gpu):
    rm.set_cuda_active(use_gpu)

    np.random.seed(45)
    v1 = np.random.rand(10, 3).astype(rm.precision)
    v2 = np.random.rand(4, 3).astype(rm.precision)
    layer = rmg.Dense(4)
    t1 = np.random.rand(10, 4).astype(rm.precision)
    t2 = np.random.rand(4, 4).astype(rm.precision)
    loss = rmg.MeanSquared()
    opt = rmg.Sgd()
    data, target = rmg.DataInput([v1, t1]).index().batch(2).get_output_graphs()
    data_t, target_t = rmg.DataInput([v2, t2]).index().batch(2).get_output_graphs()
    v = rmg.Placeholder(shape=(2, 3,))
    t = rmg.Placeholder(shape=(2, 4,))
    graph = loss(layer(v), t)
    t_exe = graph.get_executor(optimizer=opt, mode='training')
    v_exe = graph.get_executor()

    def check_validation(info):
        global validation_loss
        validation_loss = np.sum(info['epoch_loss_list'])
    t_exe.register_event('Epoch-Finish', check_validation)

    t_exe.execute({v: data, t: target}, {v: data_t, t: target_t}, epochs=3)

    def add_losses(info):
        global v_loss
        epoch_loss_list = info['epoch_loss_list']
        v_loss = np.sum(epoch_loss_list)
    v_exe.register_event('Epoch-Finish', add_losses)
    v_exe.execute({v: data_t, t: target_t}, epochs=1)
    assert np.allclose(validation_loss, v_loss)


def test_validation_executor(use_gpu):
    rm.set_cuda_active(use_gpu)

    np.random.seed(45)
    v1 = np.random.rand(4, 2).astype(rm.precision)
    layer = rmg.Dense(4)
    t1 = np.random.rand(4, 4).astype(rm.precision)
    loss = rmg.MeanSquared()
    data, target = rmg.DataInput([v1, t1]).batch(2).get_output_graphs()
    v = rmg.Placeholder(shape=(2, 2,))
    t = rmg.Placeholder(shape=(2, 4,))
    exe = loss(layer(v), t).get_executor()
    losses = []

    def add_losses(info):
        epoch_loss_list = info['epoch_loss_list']
        losses.append(np.sum(epoch_loss_list))
    exe.register_event('Epoch-Finish', add_losses)
    exe.execute({v: data, t: target}, epochs=3)
    losses1 = np.array(losses.copy())
    v2, t2 = v1 * 2, t1 * 2
    data_t, target_t = rmg.DataInput([v2, t2]).batch(2).get_output_graphs()
    losses.clear()
    losses2 = exe.execute({v: data_t, t: target_t}, epochs=3)
    losses2 = np.array(losses.copy())
    assert np.allclose(losses1 * 4, losses2)


def test_step_executor(use_gpu):
    rm.set_cuda_active(use_gpu)

    np.random.seed(45)
    v1 = np.random.rand(10, 2).astype(rm.precision)
    layer = rmg.Dense(4)
    t1 = np.random.rand(10, 4).astype(rm.precision)
    loss = rmg.MeanSquared()
    #data, target = rmg.Distro(v1, t1, batch_size=2, keyword=('a', 'b')).get_output_graphs()
    data, target = rmg.DataInput([v1, t1]).index().batch(2).get_output_graphs()
    v = rmg.Placeholder(shape=(2, 2,))
    t = rmg.Placeholder(shape=(2, 4,))
    exe = loss(layer(v), t).get_executor()
    losses = []

    def add_losses(info):
        epoch_loss_list = info['epoch_loss_list']
        losses.append(np.sum(epoch_loss_list))
    exe.register_event('Epoch-Finish', add_losses)
    exe.execute({v: data, t: target}, epochs=1)
    loss1 = np.array(losses.copy())
    data_t, target_t = rmg.DataInput([v1 * 2, t1 * 2]).index().batch(2).get_output_graphs()
    loss2 = 0
    for i in range(0, 10, 2):
        v2, t2 = v1[i:i + 2] * 2, t1[i:i + 2] * 2
        # TODO: ALlow NumPy insertion
        loss2 += exe.step({v: v2, t: t2})
    assert np.allclose(loss1 * 4, loss2)


def test_inference_mode():
    v1 = np.random.rand(10, 2).astype(rm.precision)
    model = rmg.Sequential([
        rmg.Dense(3),
        rmg.Dropout(),
    ])
    x = model(v1)
    assert model.l1._prev._fwd._op._inference is False
    x.set_inference(True)
    assert model.l1._prev._fwd._op._inference is True

    model = rmg.Sequential([
        rmg.Dense(3),
        rmg.Dropout(),
    ])
    model.set_inference(True)
    x = model(v1)
    assert model.l1._prev._fwd._op._inference is True
    x.set_inference(False)
    assert model.l1._prev._fwd._op._inference is False


def test_placeholder_forward(use_gpu):
    a = 1
    b = 2
    c = 3
    X = rmg.Placeholder(shape=(1, 1))
    Y = rmg.Placeholder(shape=(1, 1))
    Z = X + Y
    Z.feed(X, a)
    Z.feed(Y, b)
    z_result = Z.forward().as_ndarray()
    Z.print_tree()
    assert z_result == a + b
    Z.feed(Y, c)
    z_result = Z.forward().as_ndarray()
    assert z_result == a + c


def test_placeholder_backward(use_gpu):
    rm.set_cuda_active(use_gpu)
    v = rmg.StaticVariable(np.random.rand(4, 3))
    D1 = rmg.Dense(6)
    x = rmg.Placeholder(shape=(4, 6))
    D2 = rmg.Dense(2)

    g1 = D1(v)
    g2 = D2(x)

    g2.feed(x, g1)
    grad = g2.backward().get_gradient(v.output)
    t = rmg.StaticVariable(np.random.rand(4, 2))
    D3 = rmg.Dense(6)
    g3 = D3(t)
    g2.feed(x, g3)
    grad = g2.backward().get_gradient(t.output)
    try:
        g2.get_gradient(v.output)
        assert False
    except Exception:
        pass


def test_updatable_mode():
    v1 = np.random.rand(10, 2).astype(rm.precision)
    model = rmg.Sequential([
        rmg.Dense(3),
        rmg.Dropout(),
    ])
    x = model(v1)
    prev = model.l0.params['w'].as_ndarray()
    x.backward()
    x.set_updatable(False)
    x.update()
    after = model.l0.params['w'].as_ndarray()
    assert np.allclose(prev, after)


def test_finalizer(use_gpu):
    rm.set_cuda_active(use_gpu)

    np.random.seed(45)
    v = np.random.rand(2, 1, 3, 4)
    layer1 = rmg.Conv(channel=2)
    res = rmg.Reshape([-1])
    layer2 = rmg.Dense(3)
    t = np.random.rand(2, 3)
    loss = rmg.MeanSquared()
    opt = rmg.Sgd()

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
    model = rmg.Sequential([
        rmg.Dense(3),
        rmg.Dense(1),
        rmg.Dense(5),
    ])
    z = model(v).as_ndarray()
    assert z.shape == (4, 5)


def test_different_optimizers(use_gpu):
    rm.set_cuda_active(use_gpu)

    np.random.seed(45)
    v = np.random.rand(1, 1)
    opt1 = rmg.Sgd(1)
    opt2 = rmg.Sgd(-1)
    dense = rmg.Dense(1, optimizer={'w': opt1, 'b': opt2})
    k = dense(v)
    k.backward()
    grad1 = k.get_gradient(dense.params['w'].output)
    grad2 = k.get_gradient(dense.params['b'].output)
    w_before = dense.params['w'].as_ndarray()
    b_before = dense.params['b'].as_ndarray()
    k.update()
    w_after = dense.params['w'].as_ndarray()
    b_after = dense.params['b'].as_ndarray()
    assert np.allclose(w_before - grad1, w_after)
    assert np.allclose(b_before + grad2, b_after)


def test_weight_decay(use_gpu):
    rm.set_cuda_active(use_gpu)

    np.random.seed(45)
    v = np.random.rand(4, 4)
    dense = rmg.Dense(3, parameter_decay={'w': rmg.L2(0.05)})
    import os
    tmp_filename = get_random_filename()
    try:
        m1 = dense(v)
        m_arr1 = m1.as_ndarray()
        dense.save(tmp_filename)
        m1.backward().update()
        w1 = dense.params['w'].as_ndarray()

        dense.load(tmp_filename)
        m2 = dense(v)
        m2.set_regularizer(rmg.L2(0.50))
        m_arr2 = m2.as_ndarray()
        m2.backward().update()
        w2 = dense.params['w'].as_ndarray()
        assert np.allclose(m_arr1, m_arr2)
        assert not np.allclose(w1, w2)
    except Exception as e:
        os.remove(tmp_filename)
        raise e
    os.remove(tmp_filename)


class noop(rmg.core.operation):
    name = 'noop'
    _vars = {'y': rmg.core.GraphMultiStorage(shape=(0,), gpus='cpu')}

    def setup(self, inputs):
        pass

    def perform(self):
        pass


@pytest.mark.parametrize('graph_nodes', [
    {'A': rmg.core.operational_element(operation=noop(), tags=['Dummy']),
        'B': rmg.core.operational_element(operation=noop(), tags=['Dummy']),
        'C': rmg.core.operational_element(operation=noop(), tags=['Dummy'])
     },
    {'A': rmg.core.UserGraph(forward_operation=noop()),
        'B': rmg.core.UserGraph(forward_operation=noop()),
        'C': rmg.core.UserGraph(forward_operation=noop())
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
    rm.graph.core.GraphMultiStorage._gpus = None

    A = rmg.core.UserGraph(forward_operation=noop(), backward_operations=[
        noop()] if A_has_back else None)
    B = rmg.core.UserGraph(forward_operation=noop(), backward_operations=[
        noop()] if B_has_back else None)
    C = rmg.core.UserGraph(forward_operation=noop(), backward_operations=[
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
        L = rmg.core.UserLossGraph(forward_operation=noop(), backward_operations=[noop()])
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


@pytest.mark.skipif(not rm.cuda.has_cuda() or rm.cuda.cuGetDeviceCount() < 2, reason='Requires GPU')
def test_share_arr():
    rm.set_cuda_active(True)

    storage = rmg.core.graph_storage.GraphMultiStorage
    init = rm.utility.initializer.GlorotUniform()
    shape = (2, 2)

    val = storage(shape=shape, gpus=[0, 1], initializer=init)
    A = val[0].new_array()
    B = val[1].new_array()
    assert not np.allclose(A, B)
    rm.cuda.ShareInitialization()

    val = storage(shape=shape, gpus=[0, 1], initializer=init)
    A = val[0].new_array()
    B = val[1].new_array()
    assert np.allclose(A, B)


@pytest.mark.parametrize('devices_to_load', [
    'cpu', 1, 2, 3, 4
])
def test_save_load(devices_to_load):
    rmg.core.GraphMultiStorage._gpus = None
    if devices_to_load != 'cpu':
        if not rm.cuda.has_cuda() or (rm.cuda.cuGetDeviceCount() < devices_to_load):
            pytest.skip()
        rm.set_cuda_active(True)
        device_list = [d for d in range(devices_to_load)]
    else:
        device_list = devices_to_load
        rm.set_cuda_active(False)

    model = rmg.Sequential([
        rmg.Dense(3),
        rmg.Dense(6),
        rmg.Dense(2),
    ])

    x = np.random.rand(5, 4)
    y1 = model(x).as_ndarray()

    tmp_filename = get_random_filename()
    model.save(tmp_filename)

    model = rmg.Sequential([
        rmg.Dense(3),
        rmg.Dense(6),
        rmg.Dense(2),
    ])
    model.load(tmp_filename, devices=device_list)
    y2 = model(x).as_ndarray()
    div = devices_to_load if devices_to_load != 'cpu' else 1
    assert np.allclose(y1, y2 / div)

    try:
        model = rmg.Sequential([
            rmg.Dense(6),
            rmg.Dense(3),
            rmg.Dense(2),
        ])
        model.load(tmp_filename)
        raise AssertionError('Model should not be able to load different shape')
    except:
        pass

    import os
    os.remove(tmp_filename)


def test_version_save_compability(use_gpu):
    pytest.skip()  # Deprecated
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
    v3_model = rmg.Sequential([
        rmg.Dense(5),
        rmg.Dense(3),
        rmg.Dense(1),
    ])
    v3_model.load(tmp_filename)
    y2 = v3_model(x).as_ndarray()
    assert np.allclose(y1, y2)

    import os
    os.remove(tmp_filename)


def test_save_serialized(use_gpu):
    pytest.skip()  # Deprecated
    rm.set_cuda_active(use_gpu)
    eps = 3
    model_v2 = rm.BatchNormalize(epsilon=eps)
    model_v3 = rmg.BatchNormalize()

    x = np.random.rand(4, 2)
    model_v2(x)
    import os

    tmp_filename = get_random_filename()
    model_v2.save(tmp_filename)
    try:
        model_v3.load(tmp_filename)
        assert model_v3.params['_epsilon'] == eps
    except Exception as e:
        os.remove(tmp_filename)
        raise e
    os.remove(tmp_filename)


def test_gradient_clipping(use_gpu):
    if rm.precision != np.float64:
        pytest.skip()
    rm.set_cuda_active(use_gpu)

    np.random.seed(30)
    v1 = np.random.rand(20, 5)
    m = rmg.Dense(2, ignore_bias=True)
    with rmg.core.with_gradient_clipping(-1e-5, 1e-5):
        y = m(v1)
    y.backward()
    before = m.params['w'].as_ndarray()
    grad = y.get_gradient(m.params['w'].output)
    y.update(optimizer=rmg.Sgd(1, 0))
    after = m.params['w'].as_ndarray()
    diff = after - before

    delta = np.ones_like(diff) * -1e-5
    assert np.allclose(diff, delta)


@pytest.mark.parametrize('ttype', [
    np.int, np.int32, np.int64,
    np.float, np.float32, np.float64
])
def test_dtype(ttype, use_gpu):
    rm.set_cuda_active(use_gpu)

    model = rmg.Sequential([
        rmg.Dense(3),
        rmg.Dense(6),
        rmg.Dense(2),
    ])

    x = np.random.rand(5, 4).astype(ttype)
    y1 = model(x)
    assert y1.as_ndarray().dtype == rm.precision


def test_pinnedmem():  # TODO
    pytest.skip()


def test_finalize():
    pytest.skip()
