import renom as rm
import numpy as np
from renom.graph.distribution import DataInput as DI

def get_rand(*shape, dtype=np.float32):
    return np.random.rand(*shape).astype(dtype)

def test_basic_numpy():
    x = get_rand(10, 4)
    D = DI(x)

    try:
        d = D.get_output_graphs()
        assert False
    except AssertionError:
        pass

    d = D.index().get_output_graphs()
    d_x = d.as_ndarray()
    for i in range(len(x)):
        assert np.allclose(x[i], d.forward().as_ndarray())

def test_basic_generator():
    class my_gen:
        def __getitem__(self, idx):
            return np.array(idx).reshape(1,1)
        def __len__(self):
            return 7
    x = my_gen()
    D = DI(x)

    try:
        d = D.get_output_graphs()
        assert False
    except AssertionError:
        pass

    d = D.index().get_output_graphs()
    for i in range(len(x)):
        assert np.allclose(x[i], d.forward().as_ndarray())
