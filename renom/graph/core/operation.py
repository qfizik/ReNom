import numpy as np
import renom as rm
import abc
import functools


class operation(abc.ABC):
    '''
          The 'implementation' of the graph.

          The operation is responsible for implementing the actual graph, disjointing the
          graph structure from the implementatin of its components. The operation should
          be implemented using a two-part system, setup and perform. Any memory allocation
          should be done during setup.

          In order to coordinate between different operations, as well as allow the graph to understand
          the operations, the class implements several variables.

          _vars:
              Any key memory regions should be defined in this dict. Memory regions defined in the class,
              but not inserted into the _vars dict will not be considered by the overlaying graph.

          produces, consumes:
              These variables are responsible for defining the relationship between different operations.
              When a UserGraph is created, the graph defines a single forward operations and up todo
              several backward operations. The graph proceeds to look into produces and consumes to find
              any relationships between the operations.

              If the forward operation defines a key value in consumes, the graph will look in the consumes
              lists of the backward operations for the same key. If found, it is assumed that the forward
              operation contains a key memory region, found in _vars, that should be updated with the
              value defined in the backward operation for the same key.

          roles:
              The roles field allows the operation to mark itself as serving special functions. When creating
              the executor, the executor uses these roles to determine important operations, such as the input
              operations or the loss operation. This in turn allows it to perform more independently, as it
              knows where the loss should be found.

    '''
    # Tuples are immutable lists.
    produces = tuple()
    consumes = tuple()
    roles = tuple()
    # Remove vars

    # Ideally we make the connection between inputs and variables more explicit
    # inputargs = namedtuple('input_values', 'y', 'a', 'b')
    # inputs = [inputargs(*d) for d in self.inputs]

    # setup functionality should be implemented as __init__
    @abc.abstractmethod
    def setup(self, inputs):
        '''
            This method is responsible for preparing memory regions, setting internal variables
            and readying other variables that could otherwise not be prepared without knowledge
            of the graph. The setup method is called at least once before perform is called, but
            may be called multiple times depending on the state of the graph.

            Setup is called at least once before any perform, but maybe be called again later
            for several reasons. Setup should therefore recalculate any values that are input
            dependent.

            The inputs argument is the inputs to the operation as defined in the above graph.
            It is implemented as a list, where the index of each element in the list is the order
            in which the previous operation was added in the above graph.

            Note: As far as possible, this methoud should be made device agnostic.
        '''
        pass

    @abc.abstractmethod
    def perform(self):
        '''
            The perform method is responsible for placing kernels on the device in case of a
            gpu-implemented operation. As far as possible, the developer should take care to make
            no allocations of memory during the perform step and instead allocate during the setup step.

            When the graph is decided and the user creates the executor, the perform methods of the graph
            are gathered into a list and performed in order, meaning that the original operation
            will be discarded.
        '''
        pass

    def get_output_signature(self):
        return self._vars

    def __repr__(self):
        return self.name + ':\n' + self._vars['y'].__repr__()

    def as_ndarray(self):
        return self._vars['y'].as_ndarray()

    def get_key(self, key):
        return self._vars.get(key, None)

    def set_alias(self, key, alias):
        self._vars[alias] = self._vars[key]

    def optimize(self):
        return True

    def finalize(self):
        pass


class StateHolder:

    def __init__(self, null_state=None):
        if null_state is None:
            null_state = {}
        self._null_state = null_state
        self._states = [null_state]
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
        return self._states[self._cur_time + 1]
