import renom as rm
import abc
import numpy as np
from .learnable_graph import learnable_graph_element
from .operation import operation
from .new_gpu import GraphMultiStorage
import h5py

class variable_input(operation):

  name = 'Variable'
  roles = [ 'variable' ]

  def __init__(self):
    val = GraphMultiStorage()
    self._vars = { 'y' : val }

  def setup(self, inputs, storage): pass

  def perform(self): pass

class graph_variable(learnable_graph_element):

  def __init__(self):
    fwd_op = variable_input()
    self._fwd_op = fwd_op
    bwd_ops = [ ]
    super().__init__(forward_operation = fwd_op, backward_operations = bwd_ops)
  
  def set_value(self, arr, gpus = None):
    assert isinstance(arr, np.ndarray)
    v = self._fwd_op.get_key('y')
    v.__init__(shape = arr.shape, gpus = gpus)
    for gpu in v.gpus:
      v[gpu].to_gpu(arr)
   


class GraphFactory(abc.ABC):

  def __init__(self):
    self.params = dict()

  @abc.abstractmethod
  def connect(self, other): pass
  
  def __call__(self, *other):
    for param in self.params:
      self.params[param].disconnect()
    return self.connect(*other)

  def get_model_children(self):
    for k, v in self.__dict__.items():
      if isinstance(v, GraphFactory):
        yield k, v

  def _get_values(self, values):
    if self.params:
      for k in self.params.keys():
          values[1][k] = self.params[k]

    serialized = getattr(self, "SERIALIZED", ())
    for name in serialized:
      if hasattr(self, name):
        values[2][name] = getattr(self, name)

    for k, v in self.get_model_children():
      childvalues = ({}, {}, {})
      v._get_values(childvalues)
      values[0][k] = childvalues
  def values(self):
    ret = ({}, {}, {})
    self._get_values(ret)
    return ret

  def flatten_values(self):
    values = self.values()
    value_list = []

    def flatten(names, values):
      value_list.append((names, values[1], values[2]))

      for name, child_values in values[0].items():
        flatten(names + (name,), child_values)

    flatten(('root',), values)
    return value_list

  def save(self, filename):
      """Save model attributes.
      For save attributes, please register attributes to the dictionary
      which is named as 'SERIALIZED'.

      Following example shows how to do it.

      Example:
          >>> import renom as rm
          >>> import numpy as np
          >>>
          >>> class MyModel(rm.Model):
          ...     SERIALIZED = ('_moving_avg', ) # Put any attributes for saving.
          ...     def __init__(self):
          ...         super(MyModel, self).__init__()
          ...         self._l1 = rm.Dense(2)
          ...         self._l2 = rm.Dense(1)
          ...         self._moving_avg = 0
          ...     def forward(self, x):
          ...         h = self._l1(x)
          ...         h = rm.relu(h)
          ...         h = self._l2(h)
          ...         self._moving_avg = np.float32(self._moving_avg*0.5 + rm.sum(h)*0.5)
          ...         return h
          ...
          >>> model = MyModel()
          >>> model(np.random.rand(12, 4))
          >>> print(model._moving_avg)
          1.95637
          >>> model.save("test.h5") # Save
          >>> model = MyModel() # Reset model object.
          >>> model.load("test.h5") # Load
          >>> print(model._moving_avg)
          1.95637

      Args:
          filename (str): File name to save model.

      """

      value_list = self.flatten_values()
      with h5py.File(filename, 'w') as f:
       values_grp = f.create_group('values')
       types_grp = f.create_group('types')

       for names, params, attrs in value_list:
         g = values_grp.create_group('.'.join(names))
         t = types_grp.create_group('.'.join(names))

         for propname, propvalue in params.items():
           propvalue = propvalue.as_ndarray()
           g[propname] = propvalue

           t[propname] = 'renom.Variable'
           t[propname + '._auto_update'] = True 


         for propname, propvalue in attrs.items():
          if isinstance(propvalue, GPUValue):
            g['__dict__.' + propname] = propvalue.new_array()
          else:
            g['__dict__.' + propname] = propvalue

  def load(self, filename, gpus = None):
      """Load saved weights to model.

      Args:
          filename (str): File name of saved model.

      Example:
          >>> model = rm.Dense(2)
          >>> model.load("model.hd5")
      """
      f = h5py.File(filename, 'r')
      values = f['values']
      types = f['types']

      names = sorted(values.keys())

      def get_attr(root, names):
          names = names.split('.')[1:]
          ret = root
          for name in names:
              ret = getattr(ret, name)
          return ret

      target = self
      for name in names:
          target = get_attr(self, name)

          values_grp = values[name]
          types_grp = types[name]

          for k, v in values_grp.items():
              v = v.value

              if k.startswith('__dict__.'):
                  obj = target
                  name = k.split(".", 1)[1]
              else:
                  obj = target.params
                  name = k

              tmp = obj[name]
              tmp.set_value(v, gpus = gpus)

      f.close()
