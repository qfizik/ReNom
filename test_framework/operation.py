import numpy as np
import renom as rm
import abc

class operation(abc.ABC):

  @abc.abstractmethod
  def setup(self, *args, **kwargs): pass

  @abc.abstractmethod
  def perform(self): pass

  @abc.abstractmethod
  def get_output_signature(self): pass


