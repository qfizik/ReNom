import numpy as np
import abc
import functools
from warnings import warn


class graph_element(abc.ABC):
    '''
      This class implements basic logic of the graph. The graph is constructed as a tree with
      references to both the previous and following elements.

      The basic tree maintains a depth variable with the rule that its depth is the maximum
      of its predecessors plus 1. If updated, all following depth values will be recalculated.
      A node has an n-to-n relationship with previous elements in the graph and next elements,
      and should not have any relationship with nodes of the same depth as itself.

      Because of the depth in the tree constructed with graph_element, there is a sense of
      direction in the tree where smaller depths come before higher depths. Any inheriting
      classes are obliged to implement the method forward which implements the idea of
      'following' this direction implicated by the depths.

      A class inheriting can use the decorator 'walk_tree' to follow this direction simply.
      The decorated methods will be called on the entire tree guaranteeing that smaller
      depths are called first. Examples of how to use walk_tree is found below.

      The classes also contains the abstract __repr__ method, which should be implemented
      in as informative a way as possible.

    '''

    def __init__(self, previous_elements=None):

        if isinstance(previous_elements, graph_element):
            previous_elements = [previous_elements]
        elif previous_elements is None:
            previous_elements = []

        self._visited = False
        depth = 0
        for prev in previous_elements:
            prev.add_next(self)
        # This should perform a copy, not a reference store.
        self._previous_elements = previous_elements.copy()
        self.depth = depth
        self._next_elements = []
        self.update_depth()

    # Rename input and next to graph-like naming
    # Only possibly?
    def add_input(self, new_input):
        if new_input not in self._previous_elements:
            self._previous_elements.append(new_input)
        else:
            warn('Attempting to add already existing input')
        new_input.add_next(self)
        self.update_depth()

    def add_next(self, new_next):
        if new_next not in self._next_elements:
            self._next_elements.append(new_next)
        else:
            warn('Attempting to add already existing next')

    def remove_input(self, prev_input):
        if prev_input in self._previous_elements:
            if self in prev_input._next_elements:
                prev_input.remove_next(self)
            self._previous_elements.remove(prev_input)
        else:
            warn('Attempting to remove non-existing input')
        self.update_depth()

    def remove_next(self, prev_next):
        if prev_next in self._next_elements:
            self._next_elements.remove(prev_next)
        else:
            warn('Attempting to remove non-existing next')

    def remove_all_inputs(self):
        for elem in self._previous_elements[::-1]:
            self.remove_input(elem)

    def update_depth(self):
        if len(self._previous_elements) == 0:
            max_depth = 0
            self.depth = 0
        else:
            max_depth = max(p.depth for p in self._previous_elements)
            self.depth = max_depth + 1

        for next_element in self._next_elements:
            next_element.update_depth()

    def __lt__(self, other):
        return self.depth < other.depth

    @abc.abstractmethod
    def forward(self):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass

    '''
        Example of a class using walk_tree:
            In [1]: import renom as rm
            In [2]: class simple_graph(rm.graph.core.graph_element.graph_element):
               ...:     def forward(self): pass
               ...:     def __repr__(self): pass
               ...:     @rm.graph.core.graph_element.graph_element.walk_tree
               ...:     def print_depths(self):
               ...:         print(self.depth)
               ...:
            In [3]: A, B = simple_graph(), simple_graph()
            In [4]: B.add_input(A)
            In [5]: B.print_depths()
            0
            1

    '''
    @staticmethod
    def walk_tree(func):
        def cleanup(self):
            '''
            This method reset all the elements in the
            tree to a non-visited state.
            '''
            if self._visited is False:
                return
            self._visited = False
            for prev in self._previous_elements:
                cleanup(prev)
            for elem in self._next_elements:
                cleanup(elem)

        def walk_func(self, func, res, *args, **kwargs):
            if self._visited is True:
                return

            self._visited = True
            assert isinstance(res, dict) or isinstance(res, list)
            for prev in self._previous_elements:
                walk_func(prev, func, res, *args, **kwargs)

            ret = func(self, *args, **kwargs)
            if ret is not None:
                if isinstance(res, dict):
                    if self.depth not in res:
                        res[self.depth] = []
                    res[self.depth].append(ret)
                else:
                    res.append(ret)

            self._next_elements.sort()
            for elem in self._next_elements:
                if elem.depth == self.depth + 1:
                    walk_func(elem, func, res, *args, **kwargs)

        @functools.wraps(func)
        def ret_func(self, *args, flatten=False, **kwargs):
            if flatten is True:
                ret = []
            else:
                ret = {}
            walk_func(self, func, ret, *args, **kwargs)
            cleanup(self)
            return ret
        return ret_func

    def detach(self):
        for elem in self._previous_elements[::-1]:
            self.remove_input(elem)
        for elem in self._next_elements[::-1]:
            elem.remove_input(self)
