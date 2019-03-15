import renom as rm


class Activation:

    def __new__(cls, namestring, *args, **kwargs):
        assert isinstance(namestring, str)
        rmg = rm.graph

        namestring = namestring.lower()

        if namestring == 'relu':
            ret = rmg.Relu(*args, **kwargs)
        elif namestring == 'elu':
            ret = rmg.Elu(*args, **kwargs)
        elif namestring == 'leaky relu':
            ret = rmg.LeakyRelu(*args, **kwargs)
        elif namestring == 'maxout':
            ret = rmg.Maxout(*args, **kwargs)
        elif namestring == 'selu':
            ret = rmg.Selu(*args, **kwargs)
        elif namestring == 'sigmoid':
            ret = rmg.Sigmoid(*args, **kwargs)
        elif namestring == 'softmax':
            ret = rmg.Softmax(*args, **kwargs)
        elif namestring == 'softplus':
            ret = rmg.Softplus(*args, **kwargs)
        elif namestring == 'tanh':
            ret = rmg.Tanh(*args, **kwargs)
        else:
            raise ValueError('Unknown name string')

        return ret
