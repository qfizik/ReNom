'''
    This is a brief overview of the core package and its modules. In general,
    these modules provide the 'engine', which implements the graph. The names
    might change to be more descriptive later.

    graph_storage:
        The graph_storage module is responsible for providing and maintaining the storage
        of the graph through the GraphMultiStorage.



'''
from .graph_storage import GraphMultiStorage
from .user_graph import UserGraph, UserLossGraph
from .graph_element import operational_element
from .operation import operation, StateHolder
from .graph_factory import GraphFactory, graph_variable
