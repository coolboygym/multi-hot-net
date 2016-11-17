import imp

import __init__
from utils import meta_loader, meta_saver

path_fea_dir = __init__.config['path_fea_dir']
path_fea_meta = __init__.config['path_fea_meta']
fea_impl = imp.load_source('fea_impl', __init__.config['path_fea_impl'])


class Feature:
    def __init__(self, name=None, ftype='feature', proc=None, data=None, space=0, rank=0, size=0):
        self.__name = name
        self.__proc = proc
        self.__ftype = ftype
        if proc is None and hasattr(fea_impl, name + '_proc'):
            self.__proc = getattr(fea_impl, name + '_proc')
        self.__data = data
        self.__space = space
        self.__rank = rank
        self.__size = size
        self.__path = path_fea_dir + name

    def get_name(self):
        return self.__name

    def get_feature_type(self):
        return self.__ftype

    def set_feature_type(self, ftype):
        self.__ftype = ftype

    def get_proc(self):
        return self.__proc

    def set_proc(self, proc):
        self.__proc = proc

    def get_space(self):
        return self.__space

    def set_space(self, space):
        self.__space = space

    def get_rank(self):
        return self.__rank

    def set_rank(self, rank):
        self.__rank = rank

    def get_size(self, size):
        self.__size = size

    def set_size(self, size):
        self.__size = size

    def process(self, **argv):
        self.__data = self.__proc(**argv)

    def get_data(self):
        return self.__data

    def set_data(self, data):
        self.__data = data

    def get_meta(self):
        fmeta = {
            'name': self.__name,
            'ftype': self.__ftype,
            'space': self.__space,
            'rank': self.__rank,
            'size': self.__size,
        }
        return fmeta

    def set_meta(self, name=None, ftype=None, space=0, rank=0, size=0):
        self.__name = name
        self.__ftype = ftype
        self.__space = space
        self.__rank = rank
        self.__size = size

    def load_meta(self, name=None):
        if name is not None:
            fmeta = meta_loader(path_fea_meta, name)
        elif self.__name is not None:
            fmeta = meta_loader(path_fea_meta, self.__name)
        else:
            print 'name not set'
            exit(0)
        self.set_meta(**fmeta)

    def dump_meta(self):
        fmeta = self.get_meta()
        meta_saver(path_fea_meta, self.__name, fmeta)

    def __str__(self):
        return str(self.get_meta())


class NumFeature(Feature):
    def __init__(self, name, ftype='num', proc=None, data=None, space=0, rank=0, size=0):
        Feature.__init__(self, name, ftype, proc, data, space, rank, size)
