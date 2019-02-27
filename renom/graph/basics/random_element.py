import numpy as np
import renom as rm
from renom.graph.core import GraphMultiStorage, operational_element, UserGraph, operation


class random_uniform(operation):

    name = 'Random Uniform (F)'
    roles = ['static']
    keyword = None

    def __init__(self, shape, min, max, num_gpus=1):
        self._shape = [int(s) for s in shape]
        self._diff = max - min
        self._lower_bound = min
        self._gpus = list(range(num_gpus))
        self._outputs = GraphMultiStorage(shape=self._shape, gpus=self._gpus)
        self._vars = {'y': self._outputs}
        self.perform()  # This is for initialization

    def setup(self, inputs):
        pass

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self._gpus):
            rm.cuda.curand_generator().rand_uniform(self._outputs[gpu])
            rm.cuda.cumul(self._outputs[gpu], self._diff, self._outputs[gpu], handle)
            rm.cuda.cuadd(self._outputs[gpu], self._lower_bound, self._outputs[gpu], handle)

    def reset(self):
        pass


class random_uniform_cpu(random_uniform):

    def perform(self):
        self._outputs['cpu'] = (np.random.rand(self._shape) * self._diff) + self._lower_bound


class RandomUniformElement(UserGraph):

    _name = 'Random Uniform Element'

    def __init__(self, shape, min=0, max=1, num_gpus=1):
        fwd_op = random_uniform(shape, min, max, num_gpus) if rm.is_cuda_active() \
            else random_uniform_cpu(shape, min, max, num_gpus)
        super().__init__(forward_operation=fwd_op)

    def __call__(self):
        pass


def rand_uniform(shape, min=0, max=1, num_gpus=1):
    return RandomUniformElement(shape, min=min, max=max, num_gpus=num_gpus)


class random_normal(operation):

    name = 'Random Normal (F)'
    roles = ['static']
    keyword = None

    def __init__(self, shape, mean, std, num_gpus=1):
        self._shape = [int(s) for s in shape]
        self._mean = mean
        self._std = std
        self._gpus = list(range(num_gpus))
        self._outputs = GraphMultiStorage(shape=self._shape, gpus=self._gpus)
        self._vars = {'y': self._outputs}
        self.perform()  # This is for initialization

    def setup(self, inputs):
        pass

    def perform(self):
        for gpu, handle in rm.cuda.RenomHandlers(self._gpus):
            rm.cuda.curand_generator().rand_normal(self._outputs[gpu], self._mean, self._std)

    def reset(self):
        pass


class random_normal_cpu(random_normal):

    def perform(self):
        self._outputs['cpu'] = np.random.randn(*self._shape) * self._std + self._mean


class RandomNormalElement(UserGraph):

    _name = 'Random Normal Element'

    def __init__(self, shape, mean=0, std=1, num_gpus=1):
        fwd_op = random_normal(shape, mean, std, num_gpus) if rm.is_cuda_active() \
            else random_normal_cpu(shape, mean, std, num_gpus)
        super().__init__(forward_operation=fwd_op)

    def __call__(self):
        raise Exception("Random")


def rand_normal(shape, mean=0, std=1, num_gpus=1):
    return RandomNormalElement(shape, mean=mean, std=std, num_gpus=num_gpus)
