import renom as rm
import numpy as np
from renom.graph.distribution import DataInput as DI
import pytest

def get_rand(*shape, dtype=np.float32):
    return np.random.rand(*shape).astype(dtype)

DATA_POINTS = 8
OUTPUT_SHAPE = 4
class my_gen:
    def __getitem__(self, idx):
        if idx >= DATA_POINTS:
            raise IndexError()
        return np.array([idx for _ in range(OUTPUT_SHAPE)]).reshape(1,4)
    def __len__(self):
        return DATA_POINTS

NP = get_rand(DATA_POINTS, OUTPUT_SHAPE)
GEN = my_gen()

@pytest.mark.parametrize('x', [NP, GEN])
def test_basic_inputs(x, use_gpu):
    rm.set_cuda_active(use_gpu)
    D = DI(x)

    try:
        d = D.get_output_graphs()
        assert False
    except AssertionError:
        pass

    d = D.index().get_output_graphs()
    for i in range(len(x)):
        print(i)
        assert np.allclose(x[i], d.forward().as_ndarray())
    assert d.forward().as_ndarray().size == 0
    try:
        d.forward()
        assert False
    except StopIteration:
        pass
