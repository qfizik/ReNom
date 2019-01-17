import numpy as np
from tqdm import tqdm
import renom as rm
from .graph_element import graph_element
from .operational_element import operational_element
from .update_graph import update_operation
from .operation import operation


class Executor:
    '''
      The Executor class is ...

      Args:
          call_list (list):
          graph_element (GraphElement):
          losses (GraphElement):
    '''

    def __init__(self, call_list, graph_inputs, losses):
        self.call_list = call_list
        self.dispatchers = graph_inputs
        self.loss = losses

    def execute(self, epochs, progress=True):
        '''
          This function executes computational graph.

          Args:
              epochs (int): Number of epochs.
              progress (bool): If True is given, the progress will be shown.
        '''
        nth_epoch = 0
        all_losses = []
        for disp in self.dispatchers:
            disp.reset()
        while(nth_epoch < epochs):
            try:
                loss = 0
                if progress:
                    bar = tqdm()
                epoch_loss_list = []
                while(True):
                    self.perform_step()
                    loss = float(self.loss[0].as_ndarray())
                    epoch_loss_list.append(loss)
                    if progress:
                        bar.set_description("epoch:{:03d} loss:{:5.3f}".format(nth_epoch, loss))
                        bar.update(1)
            except StopIteration:
                epoch_loss_list.pop(-1)
                all_losses.append(np.sum(epoch_loss_list))
                for disp in self.dispatchers:
                    disp.reset()
                if progress:
                    bar.n = bar.n - 1
                    bar.set_description(
                        "epoch:{:03d} avg-loss:{:5.3f}".format(nth_epoch, np.mean(epoch_loss_list)))
                    bar.close()
                nth_epoch += 1
        return all_losses

    def __del__(self):
        for i in range(len(self.dispatchers)):
            self.dispatchers[i] = None
        for i in range(len(self.loss)):
            self.loss[i] = None

    def perform_step(self):
        for depth in self.call_list.keys():
            for call in self.call_list[depth]:
                call()

    def set_input_data(self, data, target):
        assert len(self.dispatchers) == 2, 'This method assumes standard input methods'
        assert isinstance(data, np.ndarray) and isinstance(
            target, np.ndarray), 'The data should be given as NumPy arrays.'
        assert len(data) == len(target), 'Data and Target should have the same number of points'
        # TODO: These are magic numbers. There should be a convention for which
        # is which instead!
        self.dispatchers[0].value = data
        self.dispatchers[1].value = target


class UserGraph(graph_element):
    '''
        The UserGraph class is the main class that will be interfacing with the user.
        The purpose of the UserGraph class is to translate more abstract ideas of layers,
        activations and losses into elements that can be interpreted by the graph
        engine, using operational_element objects.

        UserGraph follows a princinple of a relationship between forward operations and
        backward operations as 1-to-n, enforcing only a single forward operation but
        allowing several backward operations to this operation.
        Once these operations are passed to __init__, UserGraph automatically converts
        these operations to operational_elements and maintains the underlying graph.

        When the graph is constructed, the user should call either getInferenceExecutor
        or getTrainingExecutor, which will gather only the relevant information from the
        operational_element graph to support the execute method, which places on the devices.
    '''

    _name = 'Undefined'

    def __init__(self, forward_operation, backward_operations=None, previous_elements=None):
        self.connected = False
        if backward_operations is None:
            backward_operations = []

        if previous_elements is not None:
            previous_elements = UserGraph._prepare_prevs(previous_elements)

        super().__init__(previous_elements=previous_elements)

        self._create_fwd_graph(forward_operation)
        self._create_bwd_graphs(backward_operations)
        self._create_update_graphs(forward_operation, backward_operations)

        if previous_elements is not None:
            self.connect(previous_elements=previous_elements)

    # Some helper functions to divide the __init__ method into smaller parts
    def _create_bwd_graphs(self, backward_operations):
        self._bwd_graphs = []
        for op in backward_operations:
            bwd_graph = operational_element(op, tags=['Backward'])
            self._bwd_graphs.append(bwd_graph)

    @staticmethod
    def _prepare_prevs(previous_elements):
        if not isinstance(previous_elements, list):
            previous_elements = [previous_elements]
        for i, prev in enumerate(previous_elements):
            assert isinstance(prev, np.ndarray) or isinstance(prev, UserGraph)
            if isinstance(prev, np.ndarray):
                previous_elements[i] = rm.graph.StaticVariable(prev)
        return previous_elements

    def _create_fwd_graph(self, forward_operation):
        assert isinstance(forward_operation, operation) or isinstance(
            forward_operation, operational_element)
        if isinstance(forward_operation, operation):
            self._fwd = operational_element(operation=forward_operation, tags=['Forward'])
        elif isinstance(forward_operation, operational_element):
            raise NotImplementedError()
        else:
            raise AttributeError('Uknown forward operation type')

    def _create_update_graphs(self, forward_operation, backward_operations):
        if isinstance(forward_operation, operation):
            assert len(backward_operations) == len(self._bwd_graphs)
            for consumed in forward_operation.consumes:
                for op_num, op in enumerate(backward_operations):
                    if consumed in op.produces:
                        upd = update_operation(consumer=forward_operation,
                                               producer=op, key=consumed)
                        upd_g = operational_element(upd, tags=['Update'])
                        upd_g.add_input(self._bwd_graphs[op_num])

    def connect(self, previous_elements):
        if self.connected is True:
            self.detach()
            assert len(self._previous_elements) == 0 and len(self._fwd._previous_elements) == 0

        if isinstance(previous_elements, UserGraph):
            previous_elements = [previous_elements]

        for elem in previous_elements:
            self.add_input(elem)
            prev_graph_input = elem.get_forward_output()
            self._fwd.add_input(prev_graph_input)

        for num, elem in enumerate(previous_elements):
            elem.connect_back(self, pos=num)
        self.connected = True
        self.simple_forward()
        return self

    def detach(self):
        self._fwd.detach()
        for graph in self._bwd_graphs:
            graph.detach()
        super().detach()

    def connect_back(self, previous_element, pos=0):
        if len(self._bwd_graphs) == 0:
            return

        backward_graph_input = previous_element.get_backward_output(pos)
        if backward_graph_input is not None:
            for graph in self._bwd_graphs:
                graph.add_input(backward_graph_input)

    def disconnect_back(self, previous_element, pos=0):
        backward_graph_input = previous_element.get_backward_output(pos)
        for graph in self._bwd_graphs:
            graph.remove_input(backward_graph_input)

    @property
    def name(self):
        return self._name

    def __call__(self, *args, **kwargs):
        return self.connect(*args, **kwargs)

    def __repr__(self):
        self.forward()
        return self._fwd.__repr__()

    def getInferenceExecutor(self):
        ins = self._fwd.gather_operations_with_role('input', flatten=True)
        lss = self._fwd.gather_operations_with_role('loss', flatten=True)
        dct = self._fwd.get_call_dict(tag='Forward')
        ret = Executor(dct, ins, lss)
        return ret

    def getTrainingExecutor(self, optimizer=None):
        if optimizer is not None:
            ups = self._bwd_graphs[0].gather_operations_with_role('update', flatten=True)
            for i in range(len(ups)):
                ups[i].set_update_op(optimizer)
                ups[i] = None  # Avoiding destruction errors

        # Find inputs (DistributorGraphelement)
        ins = self._bwd_graphs[0].gather_operations_with_role('input', flatten=True)
        # Find loss function (UserLossGraph)
        lss = self._bwd_graphs[0].gather_operations_with_role('loss', flatten=True)
        self._fwd.continue_setup()
        dct = self._bwd_graphs[0].get_call_dict()
        ret = Executor(dct, ins, lss)
        return ret

    def simple_forward(self):
        self._fwd.forward()
        return self

    def forward(self):
        self._fwd.calculate_forward()
        return self

    def optimize(self):
        pass

    def backward(self):
        if len(self._bwd_graphs[0]._previous_elements) == 0:
            rm.graph.ConstantLoss(previous_element=self)
        self._fwd.continue_forward(tag='Backward')
        return self

    def get_gradient(self, some_variable):
        assert isinstance(some_variable, rm.graph.core.GraphMultiStorage)
        search_id = id(some_variable)
        for grph in self._bwd_graphs:
            r = grph._op.get_key(search_id)
            if r is not None:
                return r
        for elem in self._previous_elements:
            r = elem.get_gradient(some_variable)
            if r is not None:
                return r
        raise AttributeError('Could not find {}'.format(search_id))

    def update(self, optimizer=None):
        if optimizer is not None:
            ups = self._bwd_graphs[0].gather_operations_with_role('update')
            for d in ups:
                for i in range(len(ups[d])):
                    ups[d][i].set_update_op(optimizer)
                    ups[d][i] = None  # Avoiding destruction errors
        self._fwd.continue_forward(tag='Update')

    def print_tree(self):
        self._fwd.print_tree()

    def get_forward_output(self):
        return self._fwd

    def get_backward_output(self, num=0):
        if len(self._bwd_graphs) == 0:
            return None
        else:
            return self._bwd_graphs[num]

    @property
    def output(self):
        return self._fwd.output

    def as_ndarray(self):
        return self._fwd.as_ndarray()


class UserLossGraph(UserGraph):
    '''
        A special case of the UserGraph where we
    '''

    def connect(self, previous_elements):
        if isinstance(previous_elements, UserGraph):
            previous_elements = [previous_elements]
        super().connect(previous_elements)
        for elem in previous_elements:
            prev = elem.get_forward_output()
            self._bwd_graphs[0].add_input(prev)
        self._bwd_graphs[0].add_input(self._fwd)
        return self
