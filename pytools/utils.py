import cPickle as pkl
import json

import __init__
from __init__ import is_str, is_none, is_list
from matrix import get_mat_format, vstack
from sklearn.metrics import log_loss, roc_auc_score

# from node import Node, Link, NoneNode, NoneLink

BACKEND = __init__.config['backend']
DTYPE = __init__.config['dtype']
if BACKEND == 'tensorflow':
    import tensorflow as tf

    if DTYPE == 'float32':
        DTYPE = tf.float32
    elif DTYPE == 'float64':
        DTYPE = tf.float64
    elif DTYPE == 'int32':
        DTYPE = tf.int32
    elif DTYPE == 'int64':
        DTYPE = tf.int64
elif BACKEND == 'numpy':
    import numpy as np

    if DTYPE == 'float32':
        DTYPE = np.float32
    elif DTYPE == 'float64':
        DTYPE = np.float64
    elif DTYPE == 'int32':
        DTYPE = np.int32
    elif DTYPE == 'int64':
        DTYPE = np.int64

MEAN = __init__.config['mean']
STDDEV = __init__.config['stddev']
MAXVAL = __init__.config['maxval']
MINVAL = __init__.config['minval']
SEED = __init__.config['seed']

path_fea_meta = __init__.config['path_fea_meta']
path_data_meta = __init__.config['path_data_meta']
path_model_meta = __init__.config['path_model_meta']


# def as_evaluator(name):
#     if is_none(name):
#         return None
#     elif name == 'auc':
#         return AUCEvaluator()
#     elif name == 'log_loss':
#         return LogLossEvaluator()
#     else:
#         return Evaluator(name)
#
#
# def as_node(name):
#     if is_none(name):
#         return NoneNode
#     else:
#         if name in __init__.nodes:
#             return Node(name, __init__.nodes[name])
#         else:
#             u = Node(name, None)
#             __init__.nodes[name] = u
#             return u
#
#
# def as_link(name, src_name, tgt_name):
#     if is_none(name) or is_none(src_name) or is_none(tgt_name):
#         return NoneLink
#     else:
#         src_node = as_node(src_name)
#         tgt_node = as_node(tgt_name)
#         return Link(name, src_node, tgt_node)


def meta_loader(path, name=None):
    try:
        metas = json.load(open(path))
    except IOError:
        metas = {}
    if name is None:
        return metas
    else:
        return metas[name]


def meta_saver(path, name, meta):
    metas = meta_loader(path)
    if name in metas:
        print name, 'exists and overwrites'
    metas[name] = meta
    json.dump(metas, open(path, 'w'), indent=4, sort_keys=True, separators=(',', ':'))


def get_format(data):
    if is_none(data):
        return None
    elif is_str(data):
        if ':' in data:
            dname, suffix = data.split(':')
        else:
            dname = data
            suffix = 'data'
        dmeta = meta_loader(path_data_meta, dname)
        if suffix == 'data':
            return dmeta['dtype']
        elif suffix == 'label':
            return dmeta['ltype']
    else:
        return get_mat_format(data)


def get_shape(data):
    if is_none(data):
        return None
    elif is_str(data):
        if ':' in data:
            dname, suffix = data.split(':')
        else:
            dname = data
            suffix = 'data'
        dmeta = meta_loader(path_data_meta, dname)
        if suffix == 'data':
            return dmeta['dshape']
        elif suffix == 'label':
            return dmeta['lshape']
    elif get_format(data) in {'array', 'csr', 'coo', 'csc', 'libsvm', 'compound'}:
        return data.shape


def init_input_units(input_data):
    if ':' in input_data:
        dname, suffix = input_data.split(':')
    else:
        dname = input_data
        suffix = 'data'
    dmeta = meta_loader(path_data_meta, dname)
    if suffix == 'data':
        dtype = dmeta['dtype']
        if dtype == 'array':
            return tf.placeholder(DTYPE)
        elif dtype in ['csr', 'coo', 'csc', 'libsvm']:
            return tf.sparse_placeholder(DTYPE)
        elif dtype == 'compound':
            sub_types = dmeta['sub_types']
            hldrs = []
            for d in sub_types:
                if d == 'array':
                    hldrs.append(tf.placeholder(DTYPE))
                elif d in ['csr', 'coo', 'csc', 'libsvm']:
                    hldrs.append(tf.sparse_placeholder(DTYPE))
            return hldrs
    elif suffix == 'label':
        ltype = dmeta['ltype']
        if ltype == 'array':
            return tf.placeholder(DTYPE)
        elif ltype in ['csr', 'coo', 'csc', 'libsvm']:
            return tf.sparse_placeholder(DTYPE)
        elif ltype == 'compound':
            print 'compound output not supported'
            exit(0)


def init_vars_meta():
    vars_meta = {'dtype': DTYPE,
                 'mean': MEAN,
                 'stddev': STDDEV,
                 'maxval': MAXVAL,
                 'minval': MINVAL}
    return vars_meta


def init_var_map(vars_meta):
    dtype = vars_meta['dtype']
    mean = vars_meta['mean']
    stddev = vars_meta['stddev']
    maxval = vars_meta['maxval']
    minval = vars_meta['minval']
    init_path = vars_meta['init_path']
    seed = vars_meta['seed']
    vs = vars_meta['vars']
    if init_path is not None:
        loaded_var_map = pkl.load(open(init_path, 'rb'))
        print 'load var map from', init_path, loaded_var_map.keys()
    var_map = {}
    for v in vs:
        vtype = v.type or dtype
        vseed = v.init.seed or seed
        if v.init.name == 'val':
            if v.init.val == 0:
                var_map[v.name] = tf.Variable(tf.zeros(v.shape, dtype=vtype))
            elif v.init.val == 1:
                var_map[v.name] = tf.Variable(tf.ones(v.shape, dtype=vtype))
            else:
                var_map[v.name] = tf.Variable(tf.onse(v.shape, dtype=vtype) * v.init.val)
        elif v.init.name == 'normal':
            vmean = v.init.mean or mean
            vstddev = v.init.stddev or stddev
            var_map[v.name] = tf.Variable(
                tf.random_normal(v.shape, mean=vmean, stddev=vstddev, dtype=vtype, seed=vseed))
        elif v.init.name == 'tnormal':
            vmean = v.init.mean or mean
            vstddev = v.init.stddev or stddev
            var_map[v.name] = tf.Variable(
                tf.truncated_normal(v.shape, mean=vmean, stddev=vstddev, dtype=vtype, seed=vseed))
        elif v.init.name == 'uniform':
            vminval = v.init.minval or minval
            vmaxval = v.init.maxval or maxval
            var_map[v.name] = tf.Variable(
                tf.random_uniform(v.shape, minval=vminval, maxval=vmaxval, dtype=vtype, seed=vseed))
        elif v.init.name == 'load':
            vpath = v.init.init_path or init_path
            vsrc = v.init.source or v.name
            if vpath != init_path:
                print 'different init path not supported'
                exit(0)
            if loaded_var_map[vsrc].shape == tuple(v.shape):
                var_map[v.name] = tf.Variable(loaded_var_map[vsrc])
            else:
                print 'inconsistent shape', v.shape, vsrc, loaded_var_map[vsrc].shape
                exit(0)
        else:
            print 'bad param', v.init.name


def coo_2_sparse_tensor(coo_data):
    row = coo_data.row
    col = coo_data.col
    ind = vstack((row, col)).transpose()
    return ind, coo_data.data, coo_data.shape


def csr_2_sparse_tensor(csr_data):
    coo_data = csr_data.to_coo()
    return coo_2_sparse_tensor(coo_data)


def mat_2_tensor(mat):
    fmt = get_format(mat)
    if fmt == 'array':
        return mat
    elif fmt == 'csr':
        return csr_2_sparse_tensor(mat)
    else:
        print fmt, 'conversion to tensor not supported'
        exit(0)


def init_feed_dict(input_data, input_units, feed_dict):
    if input_data is None:
        return
    dfmt = get_format(input_data)
    if dfmt in {'array', 'csr'}:
        feed_dict[input_units] = mat_2_tensor(input_data)
    elif dfmt == 'compound':
        dl = input_data.data_list
        for i in range(len(dl)):
            feed_dict[input_units[i]] = mat_2_tensor(dl[i])
    else:
        print dfmt, 'not supported'
        exit(0)


def embed_input_units(input_data, input_units, weight, bias):
    fmt = get_format(input_data)
    if fmt == 'array':
        return tf.matmul(input_units, weight) + bias
    elif fmt == 'csr':
        return tf.sparse_tensor_dense_matmul(input_units, weight) + bias
    elif fmt == 'compound':
        if ':' in input_data:
            dname, suffix = input_data.split(':')
        else:
            dname = input_data
            suffix = 'data'
        dmeta = meta_loader(path_data_meta, dname)
        if suffix == 'data':
            sub_types = dmeta['sub_types']
            embeds = []
            for st in sub_types:
                if st == 'array':
                    embed_i = tf.matmul(input_units, weight) + bias
                elif st == 'csr':
                    embed_i = tf.sparse_tensor_dense_matmul(input_units, weight) + bias
                else:
                    print st, 'embedding not supported'
                    exit(0)
                embeds.append(embed_i)
            return embeds
        else:
            print 'label embedding not supported'
            exit(0)


def output(loss_func, y, y_true):
    if loss_func == 'sigmoid_log_loss':
        y_pred = tf.nn.sigmoid(y)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, y_true))
    elif loss_func == 'softmax_log_loss':
        y_pred = tf.nn.softmax(y)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_true))
    elif loss_func == 'relu_mse':
        y_pred = tf.nn.relu(y)
        loss = tf.nn.l2_loss(y_pred - y_true)
    elif loss_func == 'mse':
        y_pred = y
        loss = tf.nn.l2_loss(y_pred - y_true)
    else:
        print 'loss function not supported'
        exit(0)
    return y_pred, loss


def optimize(opt_algo, learning_rate, loss):
    if opt_algo == 'adaldeta':
        return tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'padagrad':
        return tf.train.ProximalAdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'pgd':
        return tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


def eval_metric(metrics, y_true, y_pred):
    if metrics == 'log_loss':
        return log_loss(y_true=y_true, y_pred=y_pred)
    elif metrics == 'auc':
        return roc_auc_score(y_true=y_true, y_score=y_pred)
    elif is_none(metrics):
        return {}
    elif is_list(metrics):
        eval_metrics = {}
        for m in metrics:
            eval_metrics[m] = eval_metric(m, y_true, y_pred)
        return eval_metrics
    else:
        print metrics, 'not supported'
        exit(0)


def check_early_stop(round_score, mode='min', early_stop_round=None, precision=1e-4):
    if early_stop_round is None or early_stop_round <= 0:
        return False
    elif mode == 'min':
        if np.argmin(round_score) + early_stop_round > len(round_score):
            return False
        elif np.min(round_score) - round_score[-1] < precision:
            return True
    elif mode == 'max':
        if np.argmax(round_score) + early_stop_round > len(round_score):
            return False
        elif round_score[-1] - np.max(round_score) < precision:
            return True
    return False
