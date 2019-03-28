#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)
import sys

def populate_value(namespace):
    def populator(variable):
        if not hasattr(variable, '__name__'):
            raise TypeError('Populated values must be either classes or functions')
        if namespace not in sys.modules:
            __import__(namespace)
        setattr(sys.modules[namespace], variable.__name__, variable)
        return variable
    return populator

def populate_constant(namespace, name):
    def populator(variable):
        if namespace not in sys.modules:
            __import__(namespace)
        setattr(sys.modules[namespace], name, variable)
        return variable
    return populator
