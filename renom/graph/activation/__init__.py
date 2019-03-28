from .softmax_element import Softmax, softmax
from .softplus_element import Softplus, softplus
from .relu_element import Relu, relu
from .elu_element import Elu, elu
from .selu_element import Selu, selu
from .leaky_relu_element import LeakyRelu, leaky_relu
from .tanh_element import Tanh, tanh
from .maxout_element import Maxout, maxout
from .sigmoid_element import Sigmoid, sigmoid
from .generic_activation import Activation

from renom import graph
graph.Sigmoid = Sigmoid
