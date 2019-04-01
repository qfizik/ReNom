import pytest
import os
import renom



@pytest.fixture(params=[False, True])
def use_gpu(request):
    """
    Gpu switch for test.
    """
    if request.param is True and not renom.has_cuda():
        pytest.skip()
    yield request.param
    renom.graph.core.GraphMultiStorage._gpus = None
    renom.set_cuda_active(False)


@pytest.fixture(params=[1, 2])
def num_gpu(request, use_gpu):
    """
    Gpu switch for test.
    """
    if not use_gpu and request.param > 1:
        pytest.skip()
    if request.param > 1 and (not renom.has_cuda() or renom.get_device_count() < 2):
        pytest.skip()
    return request.param


@pytest.fixture(params=[False, True])
def ignore_bias(request):
    """
    Bias switch for test.
    """
    return request.param
