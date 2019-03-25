import numpy as np
import functools
from .graph_element import graph_element
import renom as rm
import time


class operational_element(graph_element):
    '''
          The lowest graph element constructed.

          operational_element requires an operation, which defines what it does. The graph consisting
          of opertional elements is supposed to be constructed and maintained by the algorithms found in
          UserGraph and should not be be constructed directly.
    '''

    _tags_to_add = []

    def __init__(self, operation, previous_elements=None, tags=None):
        super(operational_element, self).__init__(previous_elements=previous_elements)

        self.prev_inputs = None
        self._op = operation

        self._tags = []
        if tags is not None:
            self.add_tags(new_tags=tags)
        else:
            assert False

    def add_input(self, new_input):
        super().add_input(new_input)

    def add_tags(self, new_tags):
        for tag in new_tags:
            if tag not in self._tags:
                self._tags.append(tag)
        for tag in operational_element._tags_to_add:
            if tag not in self._tags:
                self._tags.append(tag)

    def check_tags(func):
        @functools.wraps(func)
        def ret_func(self, *args, tag=None, **kwargs):
            if not isinstance(tag, list):
                tag = [tag]
            if all(t in self._tags or t is None for t in tag):
                return func(self, *args, **kwargs)
        return ret_func

    @graph_element.walk_tree
    def replace_tags(self, old_tag, new_tag):
        if old_tag in self._tags:
            self._tags[self._tags.index(old_tag)] = new_tag

    # TODO: Rename gather_operations_with_tags
    @graph_element.walk_tree
    @check_tags
    def get_call_dict(self):
        if not isinstance(self._op, rm.graph.core.graph_factory.variable_input):
            return self._op

    @graph_element.walk_tree
    @check_tags
    def gather_operations_with_role(self, role):
        if role in self._op.roles:
            return self._op

    @graph_element.walk_tree
    @check_tags
    def gather_operations(self, op, tag=None):
        if isinstance(self._op, op):
            return self._op

    def inputs_changed(self):
        inputs = []
        for prev in self._previous_elements:
            prev_inp = prev.get_output()
            if prev_inp['y'] is None:
                print('{} produced bad output'.format(prev._op.name))
                raise Exception
            inputs.append(prev_inp)
        changed = False
        if self.prev_inputs is not None:
            for inp in inputs:
                if inp not in self.prev_inputs:
                    changed = True
                    break
        else:
            changed = True
        return changed

    @check_tags
    def forward(self):
        if self.inputs_changed():
            self.setup()
        self._op.perform()

    def finalize(self):
        finished = False
        while not finished:
            rets = self._smooth_iteration(flatten=True)
            finished = all(r is True for r in rets)
        self._finalize()

    @graph_element.walk_tree
    def _smooth_iteration(self):
        return self._op.optimize()

    @graph_element.walk_tree
    def _finalize(self):
        self._op.finalize()

    def calculate_forward(self, tag=None):
        for elem in self._previous_elements:
            elem.calculate_forward(tag)
        self.forward(tag=tag)

    @graph_element.walk_tree
    def continue_forward(self, tag=None):
        self.forward(tag=tag)

    def continue_setup(self, tag=None):
        self.setup(tag=tag)
        for elem in self._next_elements:
            elem.continue_setup(tag)

    @graph_element.walk_tree
    def total_setup(self, tag=None):
        self.setup(tag=tag)

    @check_tags
    def setup(self):
        if not self.inputs_changed():
            return
        inputs = [prev.get_output() for prev in self._previous_elements]
        if rm.logging_level >= 50:
            print('{time!s}: Setting up {name!s:}'.format(time=time.ctime(), name=self._op.name))
            for i, inp in enumerate(inputs):
                print('{time!s}: Input #{0:d} has shape {1!s}'.format(
                    i, inp['y'].shape, time=time.ctime()))
        self._op.setup(inputs)
        self.prev_inputs = inputs

    @graph_element.walk_tree
    def setup_all(self):
        self.setup()

    @graph_element.walk_tree
    def print_tree(self):
        print('I am a \x1b[31m{:s}\x1b[0m at depth {:d} with tags: {}'.format(
            self._op.name, self.depth, self._tags))
        print('My prevs are: \x1b[33m{}\x1b[0m'.format(
            [v._op.name for v in self._previous_elements]))

    @graph_element.walk_tree
    @check_tags
    def set_attr(self, name, val):
        setattr(self._op, name, val)

    @property
    def name(self):
        return self._op.name

    def add_next(self, new_next):
        assert isinstance(new_next, operational_element)
        super().add_next(new_next)

    @property
    def output(self):
        if self.inputs_changed():
            self.setup()
        ret = self.get_output()['y']
        return ret

    def get_output(self):
        return self._op.get_output_signature()

    def as_ndarray(self):
        return self._op.as_ndarray()

    def __repr__(self):
        return self._op.__repr__()
