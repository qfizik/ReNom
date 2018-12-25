'''
    This is a brief overview of the core package and its modules. In general,
    these modules provide the 'engine', which implements the graph. The names
    might change to be more descriptive later.

    new_gpu:
        The new_gpu module is responsible for providing and maintaining the storage
        of the graph through the GraphMultiStorage.

'''
from .new_gpu import GraphMultiStorage
from .learnable_graph import learnable_graph_element, loss_graph_element
from .graph_element import operational_element
from .operation import operation, StateHolder
from .graph_factory import GraphFactory, graph_variable
