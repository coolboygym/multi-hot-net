import tensorflow as tf

from model import TFModel
from utils import init_vars_meta, DTYPE, embed_input_units, output, optimize
from variable import Var, as_init


class LogisticRegression(TFModel):
    def __init__(self, name=None, mtype=None, eval_metric=None, path=None, X_data=None, y_data=None, vars_init=None,
                 init_path=None, l2_reg=None, opt_algo='gd', learning_rate=0.01):
        TFModel.__init__(self, name, mtype, eval_metric, path, X_data, y_data)
        self.vars_meta = init_vars_meta()
        self.vars_meta['init_path'] = init_path
        self.vars_init = vars_init
        self.init_path = init_path
        self.l2_reg = l2_reg
        self.opt_algo = opt_algo
        self.learning_rate = learning_rate
        input_dim = tf.shape(self.X)
        output_dim = tf.shape(self.y)
        if vars_init is None:
            vars_init = {'w': as_init(name='tnormal'),
                         'b': as_init(name=0)}
        self.vars_meta['vars'] = [Var('w', DTYPE, [input_dim, output_dim], vars_init['w']),
                                  Var('b', DTYPE, [output_dim], vars_init['b'])]
        self.set_meta_adon({'vars_init', 'init_path', 'l2_reg', 'opt_algo', 'learning_rate'})

    def build_graph(self):
        w = self.vars['w']
        b = self.vars['b']
        l = embed_input_units(self.X_data, self.X, w, b)
        self.y_pred, self.loss = output(self.get_eval_metric(), l, self.y)
        if self.l2_reg is not None:
            self.loss += self.l2_reg * tf.nn.l2_loss(w)
        self.optimizer = optimize(self.opt_algo, self.learning_rate, self.loss)
