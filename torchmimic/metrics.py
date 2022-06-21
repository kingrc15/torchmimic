import numpy as np

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    mean_absolute_error,
)

def cluster_acc(Y, Y_pred):
    Y_pred, Y = np.array(Y_pred, dtype=np.int64), np.array(Y, dtype=np.int64)
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    row, col = linear_sum_assignment(w.max() - w)
    return (
        sum([w[row[i], col[i]] for i in range(row.shape[0])]) * 1.0 / Y_pred.size
    ) * 100


def accuracy(Y, Y_pred):
    return (sum(Y == Y_pred) / len(Y_pred)) * 100


def f1(Y, Y_pred):
    return f1_score(Y, Y_pred)


def balanced_accuracy(Y, Y_pred):
    return balanced_accuracy_score(Y, Y_pred)


def mae(Y, Y_pred):
    return mean_absolute_error(Y, Y_pred)


class AUCROC:
    def __init__(self, type="micro"):
        self.type = type

    def __call__(self, Y, Y_pred):
        return roc_auc_score(Y, Y_pred, multi_class="ovr", average=self.type)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, _n=1):
        self.sum += val * _n
        self.cnt += _n
        self.avg = self.sum / self.cnt


class MetricMeter:
    def __init__(self, score_fn):
        super().__init__()
        self.reset()
        self.score_fn = score_fn

    def reset(self):
        self.Y = []
        self.Y_pred = []

    def update(self, Y, Y_pred):
        self.Y_pred.append(Y_pred)
        self.Y.append(Y)

    def score(self):
        self.Y = np.concatenate(self.Y, axis=0)
        self.Y_pred = np.concatenate(self.Y_pred, axis=0)
        return self.score_fn(self.Y, self.Y_pred)
