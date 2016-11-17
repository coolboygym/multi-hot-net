import numpy as np

import __init__
from matrix import str_2_array, str_2_libsvm, libsvm_matrix
from utils import meta_loader, meta_saver, get_format, get_shape

path_data_dir = __init__.config['path_data_dir']
path_data_meta = __init__.config['path_data_meta']


class Data:
    def __init__(self, name=None, data=None, dtype=None, dshape=None, labels=None, ltype=None, lshape=None,
                 sub_types=None, sub_spaces=None):
        """
        :param name: by default name refers to the path
        :param data: None, ndarray, libsvm_matrix, csr_matrix, coo_matrix, csc_matrix, compound_matrix
        :param dtype: None, 'array', 'libsvm', 'csr', 'coo', 'csc', 'compound'
        :param dshape:
        :param labels: None, ndarray, libsvm_matrix, csr_matrix, coo_matrix, csc_matrix
        :param ltype: None, 'array', 'libsvm', 'csr', 'coo', 'csc'
        :param lshape:
        :param sub_types: only access in compound
        :param sub_spaces: only access in compound
        """
        assert dtype is None or dtype in {'array', 'libsvm', 'csr', 'coo', 'csc',
                                          'compound'}, 'data format not supported'
        assert ltype is None or ltype in {'array', 'libsvm', 'csr', 'coo', 'csc'}, 'label format not supported'
        self.__name = name
        self.__data = None
        self.__dtype = None
        self.__dshape = None
        self.__labels = None
        self.__ltype = None
        self.__lshape = None
        self.__sub_types = None
        self.__sub_spaces = None
        self.set_data(data, dtype, dshape, sub_types, sub_spaces)
        self.set_label(labels, ltype, lshape)
        self.fin = None

    def get_name(self):
        return self.__name

    def get_data(self):
        return self.__data

    def set_data(self, data, dtype=None, dshape=None, sub_types=None, sub_spaces=None):
        self.__data = data
        self.__dtype = dtype or get_format(data)
        self.__dshape = dshape or get_shape(data)
        if self.__dtype == 'compound':
            self.__sub_types = sub_types
            self.__sub_spaces = sub_spaces
            if sub_types is None and data is not None:
                self.__sub_types = map(get_format, data)
            if sub_spaces is None and data is not None:
                self.__sub_spaces = map(lambda x: x.shape[1], data)

    def get_data_type(self):
        return self.__dtype

    def set_data_type(self, dtype):
        assert dtype is None or dtype in {'array', 'libsvm', 'csr', 'coo', 'csc',
                                          'compound'}, 'data format not supported'
        self.__dtype = dtype

    def get_data_shape(self):
        return self.__dshape

    def set_data_shape(self, dshape):
        self.__dshape = dshape

    def get_sub_types(self):
        assert self.__dtype == 'compound', 'only access in compound'
        return self.__sub_types

    def set_sub_types(self, sub_types):
        assert self.__dtype == 'compound', 'only access in compound'
        self.__sub_types = sub_types

    def get_sub_spaces(self):
        assert self.__dtype == 'compound', 'only access in compound'
        return self.__sub_spaces

    def set_sub_spaces(self, sub_spaces):
        assert self.__dtype == 'compound', 'only access in compound'
        self.__sub_spaces = sub_spaces

    def get_label(self):
        return self.__labels

    def set_label(self, labels, ltype=None, lshape=None):
        self.__labels = labels
        self.__ltype = ltype or get_format(labels)
        self.__lshape = lshape or get_shape(labels)

    def get_label_type(self):
        return self.__ltype

    def set_label_type(self, ltype):
        assert ltype is None or ltype in {'array', 'libsvm', 'csr', 'coo', 'csc'}, 'label format not supported'
        self.__ltype = ltype

    def get_label_shape(self):
        return self.__lshape

    def set_label_shape(self, lshape):
        self.__lshape = lshape

    def rearrange(self, indices):
        if self.__data is not None:
            self.set_data(self.__data[indices])
        if self.__labels is not None:
            self.set_label(self.__labels[indices])

    def __len__(self):
        if self.__dshape is None:
            return 0
        else:
            return self.__dshape[0]

    def __getitem__(self, item):
        assert self.__dtype != 'coo', 'coo not supported'
        new_data = Data(name=self.__name, dtype=self.__dtype)
        if self.__data is not None:
            new_data.set_data(self.__data[item])
        if self.__labels is not None:
            new_data.set_label(self.__labels[item])
        return new_data

    def get_meta(self):
        dmeta = {
            'name': self.__name,
            'dtype': self.__dtype,
            'dshape': self.__dshape,
            'ltype': self.__ltype,
            'lshape': self.__lshape,
        }
        if self.__dtype == 'compound':
            dmeta['sub_types'] = self.__sub_types
            dmeta['sub_spaces'] = self.__sub_spaces
        return dmeta

    def set_meta(self, **dmeta):
        self.__init__(**dmeta)

    def load_meta(self):
        dmeta = meta_loader(path_data_meta, self.__name)
        self.set_meta(**dmeta)

    def dump_meta(self):
        dmeta = self.get_meta()
        meta_saver(path_data_meta, self.__name, dmeta)

    def __str__(self):
        return self.get_name()

    def open(self):
        self.fin = open(path_data_dir + self.get_name(), 'r')

    def read_buffer(self, buf_size):
        line_buffer = []
        while True:
            try:
                line_buffer.append(next(self.fin))
            except StopIteration as e:
                print e
                self.fin = None
                break
            if len(line_buffer) == buf_size:
                break
        return line_buffer

    def read(self, size=-1):
        """
        format:
            (none, libsvm): ind:val ind:val ...
            (none, array): val val ...
            (libsvm, libsvm): ind ind:val ind:val ...
            (libsvm, array): ind val val ...
            (array, libsvm): val ind:val ind:val ...
            (array, array): val val val ...
            (array, libsvm): val val ...; ind:val ind:val ...
            (array, array): val val ...; val val ...
            (libsvm, libsvm): ind:val ind:val ...; ind:val ind:val ...
            (libsvm, array): ind:val ind:val ...; val val ...
        :param size:
        :return:
        """
        if self.fin is None:
            self.open()
        line_buffer = self.read_buffer(size)
        if self.__ltype is None:
            line_y = None
            line_x = line_buffer
        elif len(line_buffer) > 0 and ';' in line_buffer[0]:
            line_y, line_x = map(lambda x: x.split(';'), line_buffer)
        else:
            line_sep = map(lambda x: x.find(' '), line_buffer)
            line_y = map(lambda x: line_buffer[x][:line_sep[x]], xrange(len(line_buffer)))
            line_x = map(lambda x: line_buffer[x][line_sep[x]:], xrange(len(line_buffer)))

        if self.__ltype is None:
            labels = None
        elif self.__ltype == 'libsvm':
            labels = libsvm_matrix(tuples=map(str_2_libsvm, line_y))
        elif self.__ltype == 'array':
            labels = np.array(map(str_2_array, line_y))
        else:
            print self.__dtype, 'not supported'
            exit(0)

        if self.__dtype == 'libsvm':
            data = libsvm_matrix(tuples=map(str_2_libsvm, line_x))
        elif self.__dtype == 'array':
            data = np.array(map(str_2_array, line_x))
        else:
            print self.__dtype, 'not supported'
            exit(0)

        self.set_data(data)
        self.set_label(labels)

# def data_2_tensor(data):
#     X = data.get_data()
#     y = data.get_label()
#     if data.get_data_type() in {'array', 'csr', 'csc', 'coo', 'libsvm'}:
#         X = mat_2_tensor(data)
#     elif data.get_data_type() == 'compound':
#         X = map(mat_2_tensor, X.data_list())
#     else:
#         print data.get_data_type(), 'not supported'
#         exit(0)
#     if data.get_label_type() is None:
#         pass
#     elif data.get_label_type() in {'array', 'csr', 'csc', 'coo', 'libsvm'}:
#         y = mat_2_tensor(y)
#     else:
#         print data.get_label_type(), 'not supported'
#         exit(0)
#     # return X, y
