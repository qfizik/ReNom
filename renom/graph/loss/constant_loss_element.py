import renom as rm
from renom.graph.core import operation, operational_element, UserLossGraph, GraphMultiStorage, GraphFactory
from renom.graph.function.sum_element import sum_forward, sum_forward_cpu
import renom.utility.initializer as init


class constant_loss_backward(operation):

    name = 'Constant (B)'

    def setup(self, inputs):

        inputs = inputs[0]['y']
        gpus = inputs.gpus
        outputs = GraphMultiStorage(shape=inputs.shape, gpus=gpus, initializer=init.Constant(1))
        self._outputs = outputs
        self._vars = {'y': outputs, 'dy': outputs}

    def perform(self):
        pass


class ConstantLoss(UserLossGraph):

    is_connector_element = True

    def __init__(self, previous_element=None):
        fwd_op = sum_forward() if rm.is_cuda_active() else sum_forward_cpu()
        fwd_op.roles = ['loss']
        bwd_ops = [constant_loss_backward()]
        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_element)
        self._bwd_graphs[0].add_input(previous_element.get_forward_output())
        self._bwd_graphs[0].add_input(self._fwd)


class ConstantLossGraphElement(GraphFactory):

    def connect(self, other):
        return ConstantLoss(previous_element=other)
