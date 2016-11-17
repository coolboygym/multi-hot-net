import __init__


class Node:
    def __init__(self, name=None, obj=None):
        self.__name = name
        self.__obj = obj
        __init__.nodes[name] = self

    def get_name(self):
        return self.__name

    def get(self):
        return self.__obj

    def set(self, obj):
        self.__obj = obj

    def __str__(self):
        return self.__name


class NoneNode(Node):
    def get_name(self):
        return 'None'

    def get(self):
        return None

    def __str__(self):
        return 'None'


class Link:
    def __init__(self, name=None, src=None, tgt=None):
        self.__name = name
        self.__src = src
        self.__tgt = tgt
        __init__.links[name] = self

    def get_name(self):
        return self.__name

    def get_source(self):
        return self.__src

    def get_target(self):
        return self.__tgt

    def __str__(self):
        return self.__src.get_name() + '->' + self.__tgt.get_name()


class NoneLink(Link):
    def get_name(self):
        return 'None'

    def get_source(self):
        return NoneNode()

    def get_target(self):
        return NoneNode()

    def __str__(self):
        return 'None->None'


class VarNode(Node):
    def __init__(self, name=None, obj=None, shape=Node, init_method=None, mean=None, stddev=None, maxval=None,
                 minval=None):
        Node.__init__(self, name, obj)
        self.shape = shape
        self.init_method = init_method
        self.mean = mean
        self.stddev = stddev
        self.maxval = maxval
        self.minval = minval

    def get_meta(self):
        vmeta = {
            'name': self.get_name(),
            'shape': self.shape,
            'init_method': self.init_method
        }
        if self.mean is not None:
            vmeta['mean'] = self.mean
        if self.stddev is not None:
            vmeta['stddev'] = self.stddev
        if self.maxval is not None:
            vmeta['maxval'] = self.maxval
        if self.minval is not None:
            vmeta['minval'] = self.minval
        return vmeta

    def set_meta(self, vmeta):
        self.__init__(**vmeta)
