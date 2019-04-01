import numpy as np

import renom as rm
from renom.graph.core import operation, GraphMultiStorage, UserGraph
from renom.graph.basics.static_variable import static_value
from renom.graph import populate_graph


class placeholder_op(operation):

    name = 'Placeholder'
    roles = ['placeholder']

    def __init__(self, shape, gpus, identifier):
        shape = list(shape)
        self.identity = identifier
        self._other = None

        outs = GraphMultiStorage(shape=shape, gpus=gpus)
        self._out = outs
        self._vars = {'y': outs}
        self.gpus = gpus

    def link(self, other_op):
        '''A method for linking graphs indirectly.

        This method is used to connect the input of one graph (A) with
        the output of another (B) by placing the input area in the
        to-be-linked graphs output area.

        Doing this allows us to connect the two graphs without forcing
        the entire graph to perform another setup call since the input to
        the elements in graph A will not have changed.
        '''
        self._other = other_op
        if not hasattr(other_op, '_vars'):
            return
        vars = other_op.get_output_signature()
        ins = vars['y']
        assert ins.shape[1:] == self._out.shape[1:]
        self._out.shape = ins.shape
        self._ins = ins
        prevs = vars['y']
        vars['y'] = self._out
        if rm.is_cuda_active():
            for gpu in prevs.gpus:
                self._out[gpu] = prevs[gpu]
        else:
            self._out['cpu'] = prevs['cpu']

    def setup(self, inputs):
        '''Placeholder setup method

        In case the original, to-be-linked graph was not yet setup,
        such as what would happen if one tries to link backward graphs
        together before any backward calls, this method performs the
        linking process during the setup.
        '''
        self.link(self._other)

    def perform(self):
        pass


@populate_graph
class Placeholder(UserGraph):
    '''A simple placeholder object.

    This graph tries to allow establishing connections between graphs,
    by forcing the output of what is it to replace with its own output.

    To use the placeholder object, either use the UserGraph.feed method
    or give the executor a feed_dict with this as a key.

    Args:
        Shape(Iterative): The shape of the Placeholder object. This is
        used by the placeholder to feed dummy data to the graph, so that
        following elements will know what to connect with.
        Note that the shape expects a batch size as well, as the batch
        size is necessary to determine the maximum required memory for
        each operation.
        num_gpus(int): The number of gpus to spread the Placeholder
        object on. If Cuda has not been activated, the argument is
        ignored and the object is placed on the CPU instead.

    Note:
        Even if a new element is fed to the placeholder through
        the feed method, the previous connections still apply.

    Example:
        >>> import numpy as np
        >>> import renom.graph as rmg
        >>> X = rmg.Placeholder(shape=(2,3))
        >>> D = rmg.Dense(5)
        >>> d = D(X)
        >>> d.print_tree()
        I am a Placeholder at depth 0 with tags: ['Forward']
        My prevs are: []
        I am a Variable at depth 0 with tags: ['Forward']
        My prevs are: []
        I am a Dense (F) at depth 1 with tags: ['Forward', 140550403884648]
        My prevs are: ['Placeholder', 'Variable']
        I am a Variable at depth 0 with tags: ['Forward']
        My prevs are: []
        I am a Bias (F) at depth 2 with tags: ['Forward', 140550403884648, 140550403884648]
        My prevs are: ['Dense (F)', 'Variable']
        >>> K = rmg.Dense(3)
        >>> k = K(np.random.rand(2,4))
        >>> d.feed(X, k)
        >>> d.print_tree()
        I am a Static Variable at depth 0 with tags: ['Forward', 140550403886776]
        My prevs are: []
        I am a Variable at depth 0 with tags: ['Forward']
        My prevs are: []
        I am a Dense (F) at depth 1 with tags: ['Forward', 140550403886776]
        My prevs are: ['Static Variable', 'Variable']
        I am a Variable at depth 0 with tags: ['Forward']
        My prevs are: []
        I am a Bias (F) at depth 2 with tags: ['Forward', 140550403886776, 140550403886776]
        My prevs are: ['Dense (F)', 'Variable']
        I am a Placeholder at depth 3 with tags: ['Forward']
        My prevs are: ['Bias (F)']
        I am a Variable at depth 0 with tags: ['Forward']
        My prevs are: []
        I am a Dense (F) at depth 4 with tags: ['Forward', 140550403884648]
        My prevs are: ['Placeholder', 'Variable']
        I am a Variable at depth 0 with tags: ['Forward']
        My prevs are: []
        I am a Bias (F) at depth 5 with tags: ['Forward', 140550403884648, 140550403884648]
        My prevs are: ['Dense (F)', 'Variable']
        >>>

    '''

    def __init__(self, shape, num_gpus=1):
        if rm.is_cuda_active():
            gpus = [gpu for gpu in range(num_gpus)]
        else:
            gpus = 'cpu'
        fwd_op = placeholder_op(shape, gpus, id(self))
        bwd_ops = [placeholder_op(shape, gpus, id(self))]
        super().__init__(fwd_op, bwd_ops)
