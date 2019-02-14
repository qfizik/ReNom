import numpy as np
import renom as rm
import renom.graph as rmg
import renom.graph.basics as basic
import renom.graph.activation as activation
import renom.graph.function as function
import renom.graph.loss as loss
import pytest


@pytest.mark.parametrize("oper", [
    basic.add_element.add_forward,
    basic.add_element.add_backward,
    basic.concatenate_element.concatenate_forward,
    basic.concatenate_element.concatenate_backward,
    basic.div_element.div_forward,
    basic.div_element.div_backward,
    basic.exp_element.exp_forward,
    basic.exp_element.exp_backward,
    basic.get_item_element.get_item_forward,
    basic.get_item_element.get_item_backward,
    basic.log_element.log_forward,
    basic.log_element.log_backward,
    basic.max_element.max_forward,
    basic.max_element.max_backward,
    basic.mean_element.mean_forward,
    basic.mean_element.mean_backward,
    basic.min_element.min_forward,
    basic.min_element.min_backward,
    basic.mul_element.mul_forward,
    basic.mul_element.mul_backward,
    basic.pow_element.pow_forward,
    basic.pow_element.pow_backward,
    basic.random_element.random_uniform,
    basic.random_element.random_normal,
    basic.reshape_element.reshape_forward,
    basic.reshape_element.reshape_backward,
    basic.sqrt_element.sqrt_forward,
    basic.sqrt_element.sqrt_backward,
    basic.square_element.square_forward,
    basic.square_element.square_backward,
    basic.static_variable.static_value,
    basic.sub_element.sub_forward,
    basic.sub_element.sub_backward,
    basic.sum_element.sum_forward,
    basic.sum_element.sum_backward,

    activation.elu_element.elu_forward,
    activation.elu_element.elu_backward,
    activation.relu_element.relu_forward,
    activation.relu_element.relu_backward,
    activation.leaky_relu_element.leaky_relu_forward,
    activation.leaky_relu_element.leaky_relu_backward,
    activation.maxout_element.maxout_forward,
    activation.maxout_element.maxout_backward,
    activation.selu_element.selu_forward,
    activation.selu_element.selu_backward,
    activation.sigmoid_element.sigmoid_forward,
    activation.sigmoid_element.sigmoid_backward,
    activation.softmax_element.softmax_forward,
    activation.softmax_element.softmax_backward,
    activation.softplus_element.softplus_forward,
    activation.softplus_element.softplus_backward,
    activation.tanh_element.tanh_forward,
    activation.tanh_element.tanh_backward,

    function.batch_normalize_element.batch_norm_forward,
    function.batch_normalize_element.batch_norm_backward,
    function.bias_element.bias_forward,
    function.bias_element.bias_backward,
    function.convolutional_element.conv_forward,
    function.convolutional_element.conv_backward,
    function.deconvolutional_element.deconv_forward,
    function.deconvolutional_element.deconv_backward,
    function.dense_element.dense_backward,
    function.dense_element.dense_forward,
    function.dropout_element.dropout_forward,
    function.dropout_element.dropout_backward,
    function.embedding_element.embedding_forward,
    function.embedding_element.embedding_weight_backward,
    function.gru_element.gru_forward,
    function.gru_element.gru_backward,
    function.l2_norm_element.l2norm_forward,
    function.l2_norm_element.l2norm_backward,
    function.layer_normalize_element.layer_norm_forward,
    function.layer_normalize_element.layer_norm_backward,
    function.lrn_element.lrn_forward,
    function.lrn_element.lrn_backward,
    function.lstm_element.lstm_forward,
    function.lstm_element.lstm_backward,
    function.pool_element.pool_forward,
    function.pool_element.pool_backward,
    function.unpool_element.unpool_forward,
    function.unpool_element.unpool_backward,
    function.weight_normalize_element.weight_norm_forward,
    function.weight_normalize_element.weight_norm_backward,

    loss.constant_loss_element.constant_loss_backward,
    loss.cross_entropy_element.cross_entropy_forward,
    loss.cross_entropy_element.cross_entropy_backward,
    loss.mean_squared_element.mean_squared_forward,
    loss.mean_squared_element.mean_squared_backward,
    loss.sigmoid_cross_entropy_element.sigmoid_forward,
    loss.sigmoid_cross_entropy_element.sigmoid_backward,
    loss.smoothed_l1_element.smoothed_l1_forward,
    loss.smoothed_l1_element.smoothed_l1_backward,
    loss.softmax_cross_entropy_element.softmax_cross_entropy_forward,
    loss.softmax_cross_entropy_element.softmax_cross_entropy_backward,
])
def test_operation_name(oper):
    name = getattr(oper, 'name', False)
    print(name)
    assert name


@pytest.mark.parametrize("layer,params,shape", [
    [function.convolutional_element.Conv, ['w'], (1, 3, 3, 3)],
    [function.deconvolutional_element.Deconv, ['w'], (1, 3, 3, 3)],
    [function.dense_element.Dense, ['w'], (1, 3)],
    [function.lstm_element.Lstm, ['w', 'wr'], (1, 3)],
    [function.gru_element.Gru, ['w', 'wr'], (1, 3)],
    [function.weight_normalize_element.WeightNormalize, ['w'], (1, 3)],
])
def test_initializer(layer, params, shape):
    init = rm.utility.initializer.Constant(-9999)
    val = rmg.StaticVariable(np.random.rand(*shape))
    graph = layer(initializer=init, ignore_bias=True)
    out = graph(val)
    operation = out._fwd._op
    for p in params:
        assert np.all(operation.get_key(p).as_ndarray() == -9999)


@pytest.mark.parametrize("layer,params,shape", [
    [function.convolutional_element.Conv, ['w'], (1, 3, 3, 3)],
    [function.deconvolutional_element.Deconv, ['w'], (1, 3, 3, 3)],
    [function.dense_element.Dense, ['w'], (1, 3)],
    [function.lstm_element.Lstm, ['w', 'wr'], (1, 3)],
    [function.gru_element.Gru, ['w', 'wr'], (1, 3)],
    [function.weight_normalize_element.WeightNormalize, ['w'], (1, 3)],
])
def test_weight_dacay(layer, params, shape):
    decay_rate = 0.1
    val = rmg.StaticVariable(np.random.rand(*shape))
    graph = layer(weight_decay=decay_rate, ignore_bias=True)
    l = rmg.ConstantLoss()
    out = graph(val)
    loss = l(out)
    operation = out._fwd._op

    for p in params:
        bkw = loss.backward()
        w = operation.get_key(p)
        grad = bkw.get_gradient(w).as_ndarray()
        prev_w = w.as_ndarray()
        loss.update()
        next_w = w.as_ndarray()
        diff1 = -(next_w - prev_w)
        diff2 = grad + decay_rate * prev_w
        loss.print_tree()
        assert np.allclose(diff1, diff2)
