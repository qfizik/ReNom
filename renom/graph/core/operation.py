import numpy as np
import renom as rm
import abc
import functools

class StateHolder:

  def __init__(self, null_state = None):
    if null_state is None:
      null_state = {}
    self._null_state = null_state
    self._states = [ null_state ]
    self._cur_time = 0

  def get_prev(self, key):
    state = self._states[self._cur_time]
    if key in state:
      return state[key]
    else:
      return self._null_state[key]

  def set_prev(self, key, val):
    self._states[self._cur_time][key] = val

  def register(self, key, val):
    self._null_state[key] = val
  
  def peek(self):
    return {**self._null_state, **self._states[self._cur_time]}

  def push(self, state):
    self._states.append(state)
    self._cur_time += 1

  def pop(self):
    self._cur_time -= 1
    return self._states[self._cur_time+1]


class operation(abc.ABC):

  produces = []
  consumes = []
  _vars = { 'y' : None }
  ready = False


  @abc.abstractmethod
  def setup(self, inputs, storage): pass

  @abc.abstractmethod
  def perform(self): pass

  def get_output_signature(self): return self._vars
  def __repr__(self): return self._vars['y'].__repr__()
  def as_ndarray(self): return self._vars['y'].as_ndarray()
  def get_key(self, key): return self._vars.get(key, None)
  def set_alias(self, key, alias): self._vars[alias] = self._vars[key]

