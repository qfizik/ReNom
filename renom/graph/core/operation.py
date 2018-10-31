import numpy as np
import renom as rm
import abc
import functools

def assert_initialized(func):
  @functools.wraps(func)
  def ret_func(self, *args, **kwargs):
    assert self._vars is not None
    func(self, *args, **kwargs)
  return ret_func

class operation(abc.ABC):

  produces = []
  consumes = []
  _funcs = []
  _args = []
  _kwargs = []


  @abc.abstractmethod
  def setup(self, inputs, storage): pass

  def register_function(self, func, *args, **kwargs):
    self._funcs.append(func)
    self._args.append(args)
    self._kwargs.append(kwargs)

  @abc.abstractmethod
  def perform(self): pass

  #@assert_initialized
  def get_output_signature(self): return self._vars
  #@assert_initialized
  def __repr__(self): return self._vars['y'].__repr__()
  #@assert_initialized
  def as_ndarray(self): return self._vars['y'].as_ndarray()
  #@assert_initialized
  def get_key(self, key): return self._vars[key]

