import pytest
import renom as rm
from renom  .utility.initializer import *
import numpy as np
from functools import partial
from renom.config import precision

##############################################################
# 1. setting model
##############################################################


class MyModel(rm.Model):
    def __init__(self):
        self.c1=rm.Conv2d(32, filter=3, padding = 1, stride=1)
        self.c2=rm.Conv2d(32, filter=3, padding = 1, stride=1)
        self.l1=rm.Dense(128)
        self.l2=rm.Dense(4)

    def forward(self,x):
        h = self.c1(x)
        h = rm.relu(h)
        h = self.c2(h)
        h = rm.relu(h)
        h = rm.flatten(h)
        h = self.l1(h)
        h = rm.relu(h)
        return self.l2(h)

##############################################################
# 2. decorator
##############################################################
def _inject(cls, names):
    @pytest.fixture(autouse=True)
    def auto_injector_fixture(self, request):
        for name in names:
            setattr(self, name, request.getfixturevalue(name))

    cls.__auto_injector_fixture = auto_injector_fixture
    return cls


def auto_inject_fixtures(*names):
    return partial(_inject, names=names)


##############################################################
# 3. fixture
##############################################################
@pytest.fixture()
def model1():
    model = MyModel()
    yield model

##############################################################
# 4. test
##############################################################


@auto_inject_fixtures('model1')
class Test_Initializer:

    # for quick implementation
    # Its better to decide margin based on statistics
    # However, implementer needs more study for this part.
    # This will be temporary
    mean_margin = 0.4  # bias
    std_margin = 1.4  # multiples

    def get_weight(self, initial_object):
        # create model and get weight
        model = self.model1
        model.set_initializer(initial_object)
        x = np.random.random((1 ,128, 128, 1))
        _ = model(x)

        weights=[]
        for w in model.iter_models():
            if hasattr(w, "_initializer"):
                weights.append(w.params.w.as_ndarray())

        return weights

    def glorot_range(self, shape, coef):
        # create glorot lim or std
        if len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) == 4:
            size = np.prod(shape[2:])
            fan_in = shape[0] * size
            fan_out = shape[1] * size
        return np.sqrt(coef / (fan_in + fan_out))

    def he_range(self, shape, coef):
        # create he lim or std
        if len(shape) == 2:
            fan_in = shape[0]
        elif len(shape) == 4:
            size = np.prod(shape[2:])
            fan_in = shape[1] * size
        return np.sqrt(coef / fan_in)

    ############################
    # the following are tests
    ############################
    def test_set_initializer(self):
        assert hasattr(self.model1, "set_initializer")

    def test_constant(self):

        weights = self.get_weight(Constant(1))

        for weight in weights:
            assert np.all(weight == np.ones_like(weight))

    def test_glorotUniform(self):

        weights = self.get_weight(GlorotUniform())

        for weight in weights:
            weight = np.abs(weight)
            shape = weight.shape
            lim = self.glorot_range(shape, 6)

            assert np.all(weight <= lim)

    def test_glorotNormal(self):

        weights = self.get_weight(GlorotNormal())

        for weight in weights:
            shape = weight.shape
            std = self.glorot_range(shape, 2)*self.std_margin

            assert np.all(np.std(weight) <= std)

    def test_heUniform(self):

        weights = self.get_weight(HeUniform())

        for weight in weights:
            weight = np.abs(weight)
            shape = weight.shape
            lim = self.he_range(shape, 6)

            assert np.all(weight <= lim)

    def test_heNormal(self):

        weights = self.get_weight(HeNormal())

        for weight in weights:
            shape = weight.shape
            std = self.he_range(shape, 2)*self.std_margin

            assert np.all(np.std(weight) <= std)

    def test_gaussian(self):

        weights = self.get_weight(Gaussian(mean=2.0, std=1.0))

        for weight in weights:
            weight -= 2

            assert np.all(np.mean(weight) <= self.mean_margin) \
                and np.all(np.std(weight) <= self.std_margin)  # margin

    def test_uniform(self):

        weights = self.get_weight(Uniform(min=-2.0, max=0.5))

        for weight in weights:
            assert np.all(weight <= 0.5) and np.all(weight >= -2)  # margin

    def test_orthogonal(self):

        weights = self.get_weight(Orthogonal())

        for weight in weights:

            weight = weight.reshape((weight.shape[0], -1))

            res1 = np.around(np.dot(weight.T, weight), decimals=2)
            res2 = np.around(np.dot(weight, weight.T), decimals=2)
            zero_count_1 = np.count_nonzero(res1 - np.diag(np.diagonal(res1)))
            zero_count_2 = np.count_nonzero(res2 - np.diag(np.diagonal(res2)))

            assert not zero_count_1 or not zero_count_2, "Not orthogonal"
