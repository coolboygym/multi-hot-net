import cPickle as pkl

import __init__
from utils import meta_loader, meta_saver, init_input_units, init_var_map, init_feed_dict

if __init__.config['backend'] == 'tensorflow':
    import tensorflow as tf

path_model_dir = __init__.config['path_model_dir']
path_model_meta = __init__.config['path_model_meta']


class Model:
    def __init__(self, name=None, mtype=None, backend=None, eval_metric=None, path=None,
                 X_data=None, Xs_data=None, y_data=None, ys_data=None):
        """
        :param name: model name/path, named as data_model_version
        :param mtype: lr, fm
        :param backend: xgb, tensorflow
        :param eval_metric: eval_metric, some can be loss function
        :param path: model dir
        :param X_data: name of input data
        :param Xs_data: name of inputs data
        :param y_data: name of input labels
        :param ys_data: name of inputs labels
            X, y could be data_name, data_name:data, or data_name:label.
            data_name will be interpreted as data_name:data
        """
        self.__name = name
        self.__mtype = mtype
        self.__backend = backend
        self.__eval_metric = eval_metric
        self.__path = path or path_model_dir + name
        self.X_data = X_data
        self.Xs_data = Xs_data
        self.y_data = y_data
        self.ys_data = ys_data
        self.X = None
        self.Xs = None
        self.y = None
        self.ys = None
        self.__meta_adon = None

    def get_name(self):
        return self.__name

    def get_model_type(self):
        return self.__mtype

    def get_backend(self):
        return self.__backend

    def set_backend(self, backend):
        self.__backend = backend

    def get_path(self):
        return self.__path

    def set_path(self, path):
        self.__path = path

    def get_eval_metric(self):
        return self.__eval_metric

    def get_meta_adon(self):
        return self.__meta_adon

    def set_meta_adon(self, meta_adon):
        self.__meta_adon = meta_adon

    def get_meta(self):
        mmeta = {
            'name': self.__name,
            'mtype': self.__mtype,
            'backend': self.__backend,
            'eval_metric': self.__eval_metric,
            'path': self.__path,
            'X_data': self.X_data,
            'Xs_data': self.Xs_data,
            'y_data': self.y_data,
            'ys_data': self.ys_data,
        }
        prefix = '_' + self.__class__.__name__ + '__'
        for m in self.__meta_adon:
            mmeta[m] = getattr(self, prefix + m)
        return mmeta

    def set_meta(self, **mmeta):
        self.__init__(**mmeta)

    def load_meta(self):
        mmeta = meta_loader(path_model_dir, self.__name)
        self.set_meta(**mmeta)

    def dump_meta(self):
        mmeta = self.get_meta()
        meta_saver(path_model_dir, self.__name, mmeta)

    def write_log(self, log_str, mode='a'):
        with open(self.__path + '.log', mode) as fout:
            fout.write(log_str)

    def init(self, **kwargs):
        pass

    def train(self, data, **kwargs):
        pass

    def predict(self, data, **kwargs):
        pass

    def dump(self, **kwargs):
        pass


class TFModel(Model):
    def __init__(self, name=None, mtype=None, backend='tensorflow', eval_metric=None, path=None,
                 X_data=None, Xs_data=None, y_data=None, ys_data=None):
        Model.__init__(self, name, mtype, backend, eval_metric, path, X_data, Xs_data, y_data, ys_data)
        self.__graph = None
        self.__sess = None
        self.__optimizer = None
        self.vars = None
        self.vars_meta = None
        self.feed_dict = None
        self.y_pred = None
        self.loss = None
        self.optimizer = None

    def __del__(self):
        self.__sess.close()

    def build_graph(self):
        pass

    def compile(self):
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            self.X = init_input_units(self.X_data)
            self.Xs = init_input_units(self.Xs_data)
            self.y = init_input_units(self.y_data)
            self.ys = init_input_units(self.ys_data)
            self.vars = init_var_map(self.vars_meta)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.__sess = tf.Session(config=config)
            self.build_graph()
            tf.initialize_all_variables().run(session=self.__sess)

    def run(self, fetches, feed_dict=None):
        return self.__sess.run(fetches, feed_dict)

    def dump(self):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(self.get_path() + '.bin', 'wb'))
        print 'model dumped at', self.get_path() + '.bin'

    def prepare_train(self, data, **kwargs):
        self.feed_dict = {}
        init_feed_dict(data.get_data(), self.X, self.feed_dict)
        init_feed_dict(data.get_label(), self.y, self.feed_dict)

    def prepare_predict(self, data, **kwargs):
        self.feed_dict = {}
        init_feed_dict(data.get_data(), self.X, self.feed_dict)

    def train(self, data, **kwargs):
        self.prepare_train(data, **kwargs)
        _, loss, y_pred = self.run(fetches=[self.optimizer, self.loss, self.y_pred], feed_dict=self.feed_dict)
        return loss, y_pred

    def predict(self, data, **kwargs):
        self.prepare_predict(data, **kwargs)
        y_pred = self.run(fetches=self.y_pred, feed_dict=self.feed_dict)
        return y_pred
