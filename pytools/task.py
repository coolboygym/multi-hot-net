import __init__

from __init__ import is_data

from utils import meta_loader, meta_saver, eval_metric, mat_2_tensor, check_early_stop
from data import Data
import model_impl as mi
import numpy as np

path_data_dir = __init__.config['path_data_dir']
path_data_meta = __init__.config['path_data_meta']
path_model_dir = __init__.config['path_model_dir']
path_model_meta = __init__.config['path_model_meta']
path_task_meta = __init__.config['path_task_meta']


class Task:
    def __init__(self, dataset, model_name, version):
        self.dataset = dataset
        self.model_name = model_name
        self.version = version
        self.tag = '%s_%s_%d' % (dataset, model_name, version)

    def parse_data_path(self, data_path):
        if data_path is None:
            return self.dataset
        elif data_path is False:
            return None
        elif '&' in data_path:
            if data_path.find('&') == 0:
                dataset = self.dataset
                suffix = data_path[1:]
            else:
                dataset, suffix = data_path.split('&')
                if dataset == 'dataset':
                    dataset = self.dataset
            return dataset + suffix
        else:
            if len(data_path) == 0:
                return self.dataset
            else:
                if data_path == 'dataset':
                    return self.dataset
                else:
                    return data_path

    def load_data(self, data):
        if not is_data(data):
            path_data = self.parse_data_path(data)
            if path_data:
                data = Data(name=path_data)
                data.load_meta()
                return data
            else:
                return None
        else:
            return data

    def parse_model_name(self):
        if self.model_name.lower() in {'lr', 'logisticregression', 'logistic_regression'}:
            return getattr(mi, 'LogisticRegression')

    def load_model(self, params):
        model_constructor = self.parse_model_name()
        model = model_constructor(name=self.tag, **params)
        if params['backend'] == 'tensorflow':
            model.compile()
        return model

    def train(self, dtrain=None, dvalid=None, buffer_size=-1, batch_size=-1, num_round=1, verbose=True, save_log=True,
              early_stop_round=None, dump_model=False, dump_task=False, model_config=None, train_config=None,
              pred_config=None, watch_list=None):
        dtrain = self.load_data(dtrain)
        dvalid = self.load_data(dvalid)
        model = self.load_model(model_config)
        train_score = []
        valid_score = []
        score_metric = model_config['eval_metric']

        def check_point(round_num, loss):
            log_str = '[%d]\tloss: %f' % (round_num, loss)
            train_y, train_y_pred = self.predict(dtrain, buffer_size, batch_size, model, pred_config)
            train_metrics = eval_metric(watch_list, train_y, train_y_pred)
            train_score.append(train_metrics[score_metric])
            log_str += ''.join(map(lambda x: '\ttrain_%s: %f' % (x[0], x[1]), train_metrics.iteritems()))
            if dvalid is not None:
                valid_y, valid_y_pred = self.predict(dvalid, buffer_size, batch_size, model, pred_config)
                valid_metrics = eval_metric(watch_list, valid_y, valid_y_pred)
                valid_score.append(valid_metrics[score_metric])
                log_str += ''.join(map(lambda x: '\tvalid_%s: %f' % (x[0], x[1]), valid_metrics.iteritems()))
            if verbose:
                print log_str
            if save_log:
                model.write_log(log_str)
            if check_early_stop(valid_score, early_stop_round):
                best_iter = np.argmin(valid_score)
                print 'best iteration: [%d]\ttrain_score: %f\tvalid_score: %f' % (
                    best_iter, train_score[best_iter], valid_score[best_iter])
                return True
            return False

        if buffer_size == -1:
            dtrain.read()
            if dvalid is not None:
                dvalid.read()
            for i in range(num_round):
                if batch_size == -1:
                    loss, _ = model.train(dtrain, **train_config)
                else:
                    loss = []
                    for j in range(int(np.ceil(len(dtrain) * 1.0 / batch_size))):
                        batch_j = dtrain[batch_size * j: batch_size * (j + 1)]
                        batch_loss, _ = model.train(batch_j, **train_config)
                        loss.extend(batch_loss)
                    loss = np.mean(loss)
                if check_point(i, loss):
                    break
        else:
            for i in range(num_round):
                dtrain.open()
                loss = []
                while dtrain.fin is not None:
                    dtrain.read(buffer_size)
                    if batch_size == -1:
                        buf_loss, _ = model.train(dtrain, **train_config)
                        loss.append(buf_loss)
                    else:
                        for j in range(int(np.ceil(len(dtrain) * 1.0 / batch_size))):
                            batch_j = dtrain[batch_size * j: batch_size * (j + 1)]
                            batch_loss, _ = model.train(batch_j, **train_config)
                            loss.extend(batch_loss)
                loss = np.mean(loss)
                if check_point(i, loss):
                    break
        if dump_model:
            model.dump()
        if dump_task:
            tmeta = {
                'task': 'train',
                'dataset': self.dataset,
                'model_name': self.model_name,
                'version': self.version,
                'tag': self.tag,
                'dtrain': str(dtrain),
                'dvalid': str(dvalid),
                'buffer_size': buffer_size,
                'batch_size': batch_size,
                'num_round': num_round,
                'verbose': verbose,
                'save_log': save_log,
                'early_stop_round': early_stop_round,
                'dump_model': dump_model,
                'dump_task': dump_task,
                'model_config': model_config,
                'train_config': train_config,
                'pred_config': pred_config,
                'watch_list': watch_list
            }
            meta_saver(path_task_meta, self.tag, tmeta)

    def predict(self, data=None, buffer_size=-1, batch_size=-1, model=None, model_config=None, pred_config=None):
        data = self.load_data(data)
        model = model or self.load_model(model_config)
        if buffer_size == -1:
            if data.fin is not None:
                data.read()
            y_true = mat_2_tensor(data.get_label())
            if batch_size == -1:
                y_pred = model.predict(data, **pred_config)
            else:
                y_pred = []
                for i in range(int(np.ceil(len(data) * 1.0 / batch_size))):
                    batch_i = data[batch_size * i: batch_size * (i + 1)]
                    batch_y_pred = model.predict(batch_i, **pred_config)
                    y_pred.extend(batch_y_pred)
        else:
            data.open()
            y_true = []
            y_pred = []
            while data.fin is not None:
                data.read(buffer_size)
                y_true.extend(data.get_label())
                if batch_size == -1:
                    buf_y_pred = model.predict(data, **pred_config)
                    y_pred.extend(buf_y_pred)
                else:
                    for i in range(int(np.ceil(len(data) * 1.0 / batch_size))):
                        batch_i = data[batch_size * i: batch_size * (i + 1)]
                        batch_y_pred = model.predict(batch_i, **pred_config)
                        y_pred.extend(batch_y_pred)
        return y_true, y_pred
