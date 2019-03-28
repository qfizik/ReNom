'''
    Trying to keep imports clean in the graph module.

    By using star imports only at the graph module level,
    we can more easily control what names are defined and where.

    All modules under the graph package should avoid star imports (for now).

    Some basic descriptions of the packages:

    basics:
        This package contains the basic classes, such as add, multiply or getitem
        classes. The main purpose of this package is to enable a NumPy like interface
        for the UserGraph class.

    function:
        The function package contains the more complicated layers used in general
        neural networks, such as the Dense or Convolutional layer.

    loss:
        Loss methods are stored in the loss package. Loss methods include loss
        functions such as Mean Squared Error or Softmax Cross Entropy,
        generally of the form L = f(x, y) where f is the loss function,
        x is the input data and y is the target value and L is the loss.

    utils:
        The utils package contains auxiliary classes and methods that provide
        support for using or speeding up the graphs. Entities that cannot be classified
        as fitting in another package belongs here. Examples are the distributor and
        sequential classes.

    activation:
        Activation classes are like functions, but generally much simpler than functions,
        taking the form y = a(x), where y is the output, a is the activation function and x
        is the input data. The difference between functions and activations is generally that
        activations have no trainable parameters.


    Please not that the core package is NOT imported into the graph module. The idea is to
    attempt to keep the internals of the graph module unexposed to the user.
'''

from renom import populate_value
populate_graph = populate_value('renom.graph')

from . import core
from . import basics
from . import train
from . import function
from . import loss
from . import utils
from . import activation
from . import distribution
