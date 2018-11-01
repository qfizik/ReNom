import numpy as np
import renom as rm
import abc
import functools

class operation(abc.ABC):

  produces = []
  consumes = []
  _vars = { 'y' : None }


  @abc.abstractmethod
  def setup(self, inputs, storage): pass

  @abc.abstractmethod
  def perform(self): pass

  def get_output_signature(self): return self._vars
  def __repr__(self): return self._vars['y'].__repr__()
  def as_ndarray(self): return self._vars['y'].as_ndarray()
  def get_key(self, key): return self._vars[key]

