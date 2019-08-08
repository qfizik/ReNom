#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import tempfile
import numpy as np
import pytest
import renom.cuda
import renom.core
from renom.cuda import set_cuda_active, use_cuda, disable_cuda, use_device
from renom.core import to_value, Variable
from renom.operation import dot, sum, sqrt, square
from renom.config import precision
from renom.layers.function.gru import Gru
import renom as rm
import test_utility
from renom.layers.function.batch_normalize import BATCH_NORMALIZE_FEATUREMAP
import itertools

# if precision is not np.float32:
#    pytestmark = pytest.mark.skip()


def rand(shape):
    return np.array(np.random.rand(*shape), dtype=precision)


def randInt(shape):
    return np.array(np.random.randint(0, 2, shape), dtype=precision)


def arange(shape):
    return np.arange(np.prod(shape), dtype=precision).reshape(shape)


def close(a, b):
    assert np.allclose(to_value(a), to_value(b), atol=1e-4, rtol=1e-3)


def close_shape(a, b):
    assert a.shape == b.shape
    return close(a, b)



@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    arange((4, 32, 3, 3)),
    arange((4, 64, 7, 7)),
])
def test_group_normalize(a):
    layer = rm.Sequential([rm.GroupNormalize()])

    set_cuda_active(True)

    g1 = Variable(a)
    g2 = layer(g1)
    g3 = rm.sum(g2)
    g = g3.grad(detach_graph=False)
    g_g1 = g.get(g1)
    g_g2 = g.get(layer.l0.params["w"])
    g_g3 = g.get(layer.l0.params["b"])

    g4 = layer(g1)

    g2.to_cpu()
    g3.to_cpu()
    g4.to_cpu()
    g_g1.to_cpu()
    g_g2.to_cpu()
    g_g3.to_cpu()

    set_cuda_active(False)

    c2 = layer(g1)
    c3 = rm.sum(c2)
    c = c3.grad(detach_graph=False)
    c_g1 = c.get(g1)
    c_g2 = c.get(layer.l0.params["w"])
    c_g3 = c.get(layer.l0.params["b"])

    c4 = layer(g1)

    close(g2, c2)
    close(g3, c3)
    close(g4, c4)
    close(c_g1, g_g1)
    close(c_g2, g_g2)
    close(c_g3, g_g3)

    close(g2.attrs._m.new_array(), c2.attrs._m)
    close(g2.attrs._v.new_array(), c2.attrs._v)

