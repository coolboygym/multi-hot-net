from sklearn.metrics import roc_auc_score, log_loss


class Evaluator:
    def __init__(self, name, eval_metric=None):
        self.__name = name
        self.__eval_metric = eval_metric

    def set_eval_metric(self, eval_metric):
        self.__eval_metric = eval_metric

    def eval(self, **argv):
        return self.__eval_metric(**argv)

    def get_name(self):
        return self.__name

    def __str__(self):
        return self.__name


class AUCEvaluator(Evaluator):
    def __init__(self):
        Evaluator.__init__(self, name='auc', eval_metric=roc_auc_score)


class LogLossEvaluator(Evaluator):
    def __init__(self):
        Evaluator.__init__(self, name='log_loss', eval_metric=log_loss)
