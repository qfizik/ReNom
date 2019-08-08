#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
　このテストファイルでは実装された自動微分により得られた勾配と
数値微分により得られた勾配を比較し、一致しているかどうかを
確認する。

　テスト時は計算精度をfloat64として実行する必要がある。
そのため、現在(2017/5/8)ではCPUにおける計算のみを
テストしている。
"""
from __future__ import division, print_function

import pytest
import warnings
import numpy as np
from renom.config import precision
import renom as rm
from renom.core import Variable
from renom.operation import sum
from renom.layers.function.group_normalize import GroupNormalize
from test_utility import auto_diff, numeric_diff

from renom.cuda import is_cuda_active, set_cuda_active, curand_generator, has_cuda
from test_utility import skipgpu

if precision is not np.float64:
    pytestmark = pytest.mark.skip()

def rand(shape):
    return np.array(np.random.rand(*shape), dtype=np.float64)


def randInteger(shape):
    return np.array(np.random.randint(0, 2, shape), dtype=np.float64)


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


def assert_cuda_active(should_be_active):
    if should_be_active is True:
        # assert has_cuda()  # Make sure we have cuda for the test
        if not has_cuda():
            warnings.warn("You are trying to use cuda but it's not installed.")
            return

    set_cuda_active(should_be_active)

    if should_be_active is True:
        assert is_cuda_active()  # Make sure it is properly activated


def compare(func, node, *args, **kwargs):
    if 'atol' in kwargs:
        atol = kwargs['atol']
    else:
        atol = 1e-5
    if 'rtol' in kwargs:
        rtol = kwargs['rtol']
    else:
        rtol = 1e-3
    ad = auto_diff(func, node, *args)
    nd = numeric_diff(func, node, *args)
    diff = ad - nd
    print("ad = \n{}".format(ad))
    print("nd = \n{}".format(nd))
    print("difference = \n{}".format(ad - nd))
    print("highest difference = \n{}".format(np.amax(ad-nd)))
    print("lowest difference = \n{}".format(np.amin(ad-nd)))
    assert np.allclose(ad, nd, atol=atol, rtol=rtol)

#--------------------------------------------------------------------------

@pytest.mark.parametrize("node,use_gpu",[
#    [Variable(rand((2,32,7,7))),True],
#    [Variable(rand((2,32,7,7))),True],
    [Variable(rand((1,64,3,3))),True],
    [Variable(rand((1,64,5,5))),False],
    [Variable(rand((1,32,7,7))),False],
])

def test_group_normalize(node, use_gpu):
    node = Variable(node)
    assert_cuda_active(use_gpu)

    layer = GroupNormalize()

    def func(node):
        return sum(layer(node))
#    compare(func, node, node)
    layer(node)
    compare(func, layer.params["w"],node)
    compare(func, layer.params["b"],node)

#--------------------------------------------------------------------------

