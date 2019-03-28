from .constant_loss_element import ConstantLossElement, ConstantLoss
from .cross_entropy_element import CrossEntropyElement, CrossEntropy
from .softmax_cross_entropy_element import SoftmaxCrossEntropyElement, SoftmaxCrossEntropy
from .sigmoid_cross_entropy_element import SigmoidCrossEntropyElement, SigmoidCrossEntropy
from .mean_squared_element import MeanSquaredElement, MeanSquared
from .smooth_l1_element import SmoothL1Element, SmoothL1

from renom import graph

graph.ConstantLossElement = ConstantLossElement
graph.ConstantLoss = ConstantLoss
graph.CrossEntropyElement = CrossEntropyElement
graph.CrossEntropy = CrossEntropy
graph.SoftmaxCrossEntropyElement = SoftmaxCrossEntropyElement
graph.SoftmaxCrossEntropy = SoftmaxCrossEntropy
graph.MeanSquaredElement = MeanSquaredElement
graph.MeanSquared = MeanSquared
graph.SmoothL1Element = SmoothL1Element
graph.SmoothL1 = SmoothL1
