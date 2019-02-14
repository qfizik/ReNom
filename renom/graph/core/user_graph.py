from numbers import Number
import numpy as np
import renom as rm
from .graph_element import graph_element
from .operational_element import operational_element
from .update_graph import update_operation
from .operation import operation
from .executor import Executor


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

        When the graph is constructed, the user should call get_executor,
        which will gather only the relevant information from the operational_element
        graph to support the execute method, which places on the devices.
    '''

    _name = 'Undefined'
    _add_clipping = None

    def __init__(self, forward_operation, backward_operations=None, previous_elements=None):
        if backward_operations is None:
            backward_operations = []

        if previous_elements is not None:
            previous_elements = UserGraph._prepare_prevs(previous_elements)

        super().__init__(previous_elements=previous_elements)

        self._create_fwd_graph(forward_operation)
        self._create_bwd_graphs(backward_operations)
        self._create_update_graphs(forward_operation, backward_operations)

        if previous_elements is not None:
            self.connect(previous_elements=previous_elements.copy())

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
            assert isinstance(prev, (np.ndarray, UserGraph, Number))
            if isinstance(prev, Number):
                previous_elements[i] = rm.graph.StaticVariable(np.array([prev]))
            elif isinstance(prev, np.ndarray):
                previous_elements[i] = rm.graph.StaticVariable(prev)
        return previous_elements

    def _create_fwd_graph(self, forward_operation):
        assert isinstance(forward_operation, operation) or isinstance(
            forward_operation, operational_element)
        if isinstance(forward_operation, operation):
            self._fwd = operational_element(operation=forward_operation, tags=['Forward'])
        elif isinstance(forward_operation, operational_element):
            assert 'Forward' in forward_operation._tags
            self._fwd = forward_operation
        else:
            raise AttributeError('Uknown forward operation type')

    def _create_update_graphs(self, forward_operation, backward_operations):
        updates = []
        if isinstance(forward_operation, operation):
            assert len(backward_operations) == len(self._bwd_graphs)
            for consumed in forward_operation.consumes:
                for op_num, op in enumerate(backward_operations):
                    if consumed in op.produces:
                        upd = update_operation(consumer=forward_operation,
                                               producer=op, key=consumed)
                        upd_g = operational_element(upd, tags=['Gradient'])
                        clipping = UserGraph._add_clipping
                        if clipping is not None:
                            if rm.is_cuda_active():
                                clip_op = rm.graph.basics.clip_element.clip_forward
                            else:
                                clip_op = rm.graph.basics.clip_element.clip_forward_cpu
                            clip = operational_element(
                                clip_op(clipping[0], clipping[1]),
                                tags=['Gradient'])
                            clip.add_input(self._bwd_graphs[op_num])
                            prv = clip
                            upd_g.add_input(prv)
                            upd_g = clip
                        else:
                            prv = self._bwd_graphs[op_num]
                            upd_g.add_input(prv)
                        updates.append(((op_num, consumed), upd_g))
        self._update_graphs = updates

    def connect(self, previous_elements):
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
        self.simple_forward()
        return self

    def detach(self):
        self._fwd.detach()
        for graph in self._bwd_graphs:
            graph.detach()
        super().detach()
        for (back_num, back_key), update in self._update_graphs:
            update.add_input(self._bwd_graphs[back_num])

    @staticmethod
    def set_gradient_clipping(use_clipping=True, floor=-1, ceil=1):
        if use_clipping is True:
            UserGraph._add_clipping = (floor, ceil)
        else:
            UserGraph._add_clipping = None

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

    def get_executor(self, mode='inference', optimizer=None, with_validation=False):
        if mode != 'inference' and optimizer is not None:
            ups = self._bwd_graphs[0].gather_operations_with_role('update', flatten=True)
            for i in range(len(ups)):
                ups[i].set_update_op(optimizer)

        fwds = self._fwd.get_call_dict(tag='Forward')
        bwds = self._fwd.get_call_dict(tag='Backward')
        grds = self._fwd.get_call_dict(tag='Gradient')
        call_list = {
            'Forward': fwds,
            'Backward': bwds,
            'Gradient': grds,
        }
        ins = self._bwd_graphs[0].gather_operations_with_role('input', flatten=True)
        ins.extend(self._fwd.gather_operations_with_role('static', flatten=True))
        # Find loss function (UserLossGraph)
        lss = self._bwd_graphs[0].gather_operations_with_role('loss', flatten=True)
        ops = {
            'graph_inputs': ins,
            'losses': lss,
            'root': self._fwd._op,
        }
        if mode == 'training':
            self._fwd.continue_setup()
        ret = Executor(call_list, ops, mode)
        if with_validation is True:
            ret._set_validation()
        return ret

    def set_inference(self, inference=True):
        if id(self) in self._fwd._tags:
            self._fwd.set_attr('_inference', inference, tag=id(self))
        else:
            assert False
            self._fwd._op._inference = inference

    def set_updatable(self, inference=True):
        if id(self) in self._fwd._tags:
            self._fwd.set_attr('_should_update', inference, tag=[id(self), 'Gradient'])
        else:
            assert False
            self._fwd._op._inference = inference

    def set_all_inference(self, inference=True):
        infs = self._fwd.gather_operations_with_role('inference', flatten=True)
        for inf in infs:
            inf._inference = inference

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
            rm.graph.ConstantLossElement(previous_element=self)
        self._fwd.continue_forward(tag='Backward')
        return self

    def get_gradient(self, some_variable):
        assert isinstance(some_variable, rm.graph.core.GraphMultiStorage)
        search_id = id(some_variable)
        backs = self._fwd.get_call_dict(tag='Backward', flatten=True)
        for b in backs:
            r = b.get_key(search_id)
            if r is not None:
                return r
        raise AttributeError('Could not find {}'.format(search_id))

    def update(self, optimizer=None):
        if optimizer is not None:
            ups = self._fwd.get_call_dict(tag='Gradient')
            for d in ups:
                for i in range(len(ups[d])):
                    if hasattr(ups[d][i], 'set_update_op'):
                        ups[d][i].set_update_op(optimizer)
        self._fwd.continue_forward(tag='Gradient')

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
