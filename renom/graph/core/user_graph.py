from numbers import Number
import numpy as np
import renom as rm
from .graph_element import graph_element
from .operational_element import operational_element
from renom.graph.core.update_graph import update_operation, gradient_accumulator
from .operation import operation
from .executor import Executor


def convertToUserGraph(to_convert):
    '''A method to convert generic objects into UserGraph objects.

    This method takes the argument to_convert and produces a UserGraph equivalent
    object that can be used to produce the value given. It is assumed that the
    value from to_convert is static, meaning that it will not change no matter how
    often it is called.

    Args:
        to_convert(np.ndarray, number): The object to be converted.
    '''
    assert isinstance(to_convert, (np.ndarray, UserGraph, Number))
    if isinstance(to_convert, Number):
        arr = np.array(to_convert, dtype=rm.precision).reshape(1, 1)
        ret = rm.graph.StaticVariable(arr)
    elif isinstance(to_convert, np.ndarray):
        ret = rm.graph.StaticVariable(to_convert)
    elif isinstance(to_convert, UserGraph):
        ret = to_convert
    else:
        raise AttributeError('Received {}'.format(type(to_convert)))
    return ret


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
        self._out_bwds = self._bwd_graphs.copy()

    @staticmethod
    def _prepare_prevs(previous_elements):
        if not isinstance(previous_elements, list):
            previous_elements = [previous_elements]
        for i, prev in enumerate(previous_elements):
            previous_elements[i] = convertToUserGraph(prev)
        return previous_elements

    def _create_fwd_graph(self, forward_operation):
        assert isinstance(forward_operation, (operation, operational_element))
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

        self.connect_forward(previous_elements)

        for num, elem in enumerate(previous_elements):
            elem.connect_back(self, pos=num)
        self.simple_forward()
        return self

    def remove_input(self, prev_input):
        super().remove_input(prev_input)

    def remove_next(self, prev_next):
        super().remove_next(prev_next)

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

    def connect_forward(self, previous_elements):
        for elem in previous_elements:
            self.add_input(elem)
            prev_graph_input = elem.get_forward_output()
            self._fwd.add_input(prev_graph_input)

    def connect_back(self, previous_element, pos=0):
        if len(self._bwd_graphs) == 0:
            return

        backward_graph_input = previous_element.get_backward_output(pos)
        if backward_graph_input is not None:
            for i, graph in enumerate(self._bwd_graphs):
                if len(graph._previous_elements) > 0 and \
                        not isinstance(graph._op, gradient_accumulator):
                    acc_op = gradient_accumulator()
                    acc_g = operational_element(acc_op, tags=['Backward'])
                    prevs = graph._previous_elements.copy()
                    graph.remove_all_inputs()
                    graph.add_input(acc_g)
                    for elem in prevs:
                        acc_g.add_input(elem)
                        graph = acc_g
                    self._bwd_graphs[i] = graph

                print(graph.name, backward_graph_input.name)
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

    def get_executor(self, mode='inference', optimizer=None):
        ret = Executor(self, mode)
        if mode != 'inference' and optimizer is not None:
            ups = self._fwd.get_call_dict(tag='Gradient')
            for d in ups:
                for i in range(len(ups[d])):
                    if hasattr(ups[d][i], 'set_update_op'):
                        ups[d][i].set_update_op(optimizer)

        if mode == 'training':
            self._fwd.total_setup()
        #ret = Executor(call_list, ops, mode)
        self._fwd.finalize()
        return ret

    def get_executor_info(self):
        '''A method used by the executor.

        Returns the graph execution list, as well as other information
        that imposes 'meaning' unto the graph, such as designating
        certain operations to be loss or input operations.

        Returns:
            call_list(dict): A dictionary that is divded into parts
            Forward, Backward and Gradient, which subsequently are divided
            into depths.
            ops(dict): Special operations in the graph, desginating certain
            operations as loss, input, etc.
        '''
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
            'root_op': self._fwd._op,
        }
        return call_list, ops

    @graph_element.walk_tree
    def feed(self, to_replace, replace_with):
        '''Replaces Placeholder objects in the graph.

        This method searches through the UserGraph-level graph and inserts replace_with
        in the location of to_replace. Note that this does not replace the original
        Placeholder objects, but simply transfers the values between the graphs.
        e.g:
            x = Placeholder (NoOp)
            L =  x -> Dense -> MeanSquared
            y = DataInput
            L.feed(x, y) = DataInput -> x -> Dense -> MeanSquared

        When using the executor, use the feed_dict argument to feed placeholder
        variables instead.

        Args:
            to_replace:
                The placeholder object to be replaced.
            replace_with:
                The UserGraph to replace it with.

        Notes:
            In the future, it may be possible to replace any UserGraph object.

        TODO:
            Add attaching for graphs with backward operations.

        '''
        assert isinstance(to_replace, rm.graph.Placeholder)
        if not isinstance(replace_with, UserGraph):
            replace_with = convertToUserGraph(replace_with)
        if to_replace is self:
            if len(self._previous_elements) > 0:
                self.remove_all_inputs()
                self._fwd.remove_all_inputs()
                bwd = self._bwd_graphs[0]
                for elem in bwd._next_elements:
                    elem.remove_input(bwd)
            self.add_input(replace_with)
            self._fwd.add_input(replace_with.get_forward_output())
            replace_with.connect_back(self, 0)
            self._fwd._op.link(replace_with.get_forward_output()._op)
            prevs = len(self._bwd_graphs[0]._previous_elements)
            assert prevs <= 1
            if prevs < 0:
                bbwd = self._bwd_graphs[0]._previous_elements[0]
                self._bwd_graphs[0]._op.link(bbwd._op)

    def set_inference(self, inference=True):
        if id(self) in self._fwd._tags:
            self._fwd.set_attr('_inference', inference, tag=id(self))
        else:
            assert False
            self._fwd._op._inference = inference

    def set_updatable(self, should_update=True):
        if id(self) in self._fwd._tags:
            self._fwd.set_attr('_should_update', should_update, tag=[id(self), 'Gradient'])
        else:
            assert False
            self._fwd._op._should_update = should_update

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

    def set_regularizer(self, regularizer):
        tags = [id(self), 'Gradient']
        for graph in self._out_bwds:
            updates = graph.get_call_dict(tag=tags, flatten=True)
            for op in updates:
                if isinstance(regularizer, dict):
                    key = op._shared_key
                    if key in regularizer:
                        op._regularizer = regularizer[key].create_op()
                else:
                    op._regularizer = regularizer.create_op()

    def set_optimizer(self, optimizer):
        tags = [id(self), 'Gradient']
        for graph in self._out_bwds:
            updates = graph.get_call_dict(tag=tags, flatten=True)
            for op in updates:
                if isinstance(optimizer, dict):
                    key = op._shared_key
                    if key in optimizer:
                        op._factory = optimizer[key]
                else:
                    op._factory = optimizer

    def backward(self):
        '''This function performs back propagation.

        Returns:
            (UserGraph): Returns object itself.
        '''
        if len(self._bwd_graphs[0]._previous_elements) == 0:
            rm.graph.ConstantLossElement(previous_element=self)
        self._fwd.continue_forward(tag='Backward')
        return self

    def get_gradient(self, variable):
        '''This function returns gradient according to given object.

        Args:
            variable (GraphMultiStorage, UserGraph): Gradient of given variable will be returned.

        Returns:
            (ndarray): Numpy ndarray.
        '''
        assert isinstance(variable, (rm.graph.core.GraphMultiStorage, UserGraph))

        if isinstance(variable, rm.graph.core.GraphMultiStorage):
            search_id = id(variable)
        elif isinstance(variable, UserGraph):
            search_id = id(variable.output)

        backs = self._fwd.get_call_dict(tag='Backward', flatten=True)
        found_grad = False
        cum_grad = 0
        for b in backs:
            r = b.get_key(search_id)
            if r is not None:
                if isinstance(r, rm.graph.core.GraphMultiStorage):
                    cum_grad += r.as_ndarray()
                elif isinstance(r, np.ndarray):
                    cum_grad += r
                found_grad = True
        if found_grad:
            return cum_grad

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
        if len(self._bwd_graphs) <= num:
            return None
        else:
            bwd_g = self._out_bwds[num]
            return bwd_g

    @property
    def output(self):
        return self._fwd.output

    def as_ndarray(self):
        '''This function returns ndarray.

        Returns:
            (ndarray): Numpy ndarray object.
        '''
        return self._fwd.as_ndarray()


class UserLossGraph(UserGraph):
    '''
        A special case of the UserGraph where we
    '''

    def connect_back(self, previous_element, pos=0):
        if len(self._bwd_graphs) == 0:
            return

        backward_graph_input = previous_element.get_backward_output(pos)
        if backward_graph_input is not None:
            for graph in self._bwd_graphs:
                graph.add_input(backward_graph_input)

    def connect(self, previous_elements):
        if isinstance(previous_elements, UserGraph):
            previous_elements = [previous_elements]
        super().connect(previous_elements)
        for elem in previous_elements:
            prev = elem.get_forward_output()
            self._bwd_graphs[0].add_input(prev)
        self._bwd_graphs[0].add_input(self._fwd)
        return self
