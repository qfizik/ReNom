'''
    This is a brief overview of the core package and its modules. In general,
    these modules provide the 'engine', which implements the graph. The names
    might change to be more descriptive later.

    graph_storage:
        The graph_storage module is responsible for providing and maintaining the storage
        of the graph through the GraphMultiStorage.

    user_graph:
        In the use_graph module, we store the UserGraph class and related classes.
        The module contents are responsible for interfacing the user with the
        underlying components.

    operational_element:
        This module comprises the basic operational_element class.
        Both UserGraph as well as operational_element are graph_elements. The UserGraph should
        be translated into operational_element components, which is what runs the graph.

    operation:
        The basic 'building blocks' of the graph is the operation class. A well constructed
        graph is a series of connected operations. When the graph executes, it performs
        the operations found in the graph.

    graph_factory:
        A class made to simplify the definitions of graph building. Instead of maintaining
        and reconnecting a single graph, the user simply constructs a new one using the
        GraphFactory. The variables are kept consistent through different graphs using
        the graph_variable.

'''
from .graph_storage import GraphMultiStorage
from .user_graph import UserGraph, UserLossGraph
from .operational_element import operational_element
from .operation import operation, StateHolder
from .graph_factory import GraphFactory, graph_variable
import contextlib as cl


@cl.contextmanager
def with_gradient_clipping(floor=None, ceil=None):
    if floor is None and ceil is None:
        UserGraph.set_gradient_clipping(False)
    else:
        if floor is None:
            floor = -2**31
        if ceil is None:
            ceil = 2**31 - 1
        UserGraph.set_gradient_clipping(True, floor, ceil)
        yield
    UserGraph.set_gradient_clipping(False)
