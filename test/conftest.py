import pytest
import renom


@pytest.fixture(params=[False, True])
def use_gpu(request):
    """
    Gpu switch for test.
    """
    if request.param is True and not renom.has_cuda():
        pytest.skip()
    return request.param


@pytest.fixture(params=[False, True])
def ignore_bias(request):
    """
    Bias switch for test.
    """
    return request.param
