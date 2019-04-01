#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

'''
    This is a brief overview of the core package and its modules. In general,
    these modules provide the ``engine``, which implements the graph. The names
    might change to be more descriptive later.

    GraphMultiStorage:
        The GraphMultiStorage module is responsible for providing and maintaining the storage
        of the graph through the GraphMultiStorage.

    GraphFactory:
        A class made to simplify the definitions of graph building. Instead of maintaining
        and reconnecting a single graph, the user simply constructs a new one using the
        GraphFactory. The variables are kept consistent through different graphs using
        the graph_variable.

    UserGraph:
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

'''

from renom.graph.core.graph_storage import GraphMultiStorage
from renom.graph.core.user_graph import UserGraph, UserLossGraph
from renom.graph.core.operational_element import operational_element
from renom.graph.core.operation import operation
from renom.graph.core.graph_factory import GraphFactory, graph_variable
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


@cl.contextmanager
def _with_operational_tag(tag):
    operational_element._tags_to_add.append(tag)
    yield
    operational_element._tags_to_add.remove(tag)
