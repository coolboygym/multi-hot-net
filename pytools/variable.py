from __init__ import is_val


class Init:
    def __init__(self, name=None):
        self.name = name


class LoadInit(Init):
    def __init__(self, name='load', init_path=None, source=None):
        Init.__init__(self, name)
        self.init_path = init_path
        self.source = source


class ValInit(Init):
    def __init__(self, name='val', val=0):
        Init.__init__(self, name)
        self.val = val


class NormalInit(Init):
    def __init__(self, name='normal', mean=None, stddev=None, seed=None):
        Init.__init__(self, name)
        self.mean = mean
        self.stddev = stddev
        self.seed = seed


class TNormalInit(Init):
    def __init__(self, name='tnormal', mean=None, stddev=None, seed=None):
        Init.__init__(self, name)
        self.mean = mean
        self.stddev = stddev
        self.seed = seed


class UniformInit(Init):
    def __init__(self, name='uniform', minval=None, maxval=None, seed=None):
        Init.__init__(self, name)
        self.minval = minval
        self.maxval = maxval
        self.seed = seed


def as_init(**kwargs):
    init_method = kwargs['name']
    if init_method == 'load':
        return LoadInit(**kwargs)
    elif init_method == 'val':
        return ValInit(**kwargs)
    elif init_method == 'normal':
        return NormalInit(**kwargs)
    elif init_method == 'tnormal':
        return TNormalInit(**kwargs)
    elif init_method == 'uniform':
        return UniformInit(**kwargs)
    elif is_val(init_method):
        kwargs['name'] = 'val'
        kwargs['val'] = init_method
        return ValInit(**kwargs)
    else:
        print 'not implemented'
        exit(0)


class Var:
    def __init__(self, name=None, type=None, shape=None, init=None):
        self.name = name
        self.type = type
        self.shape = shape
        self.init = init
