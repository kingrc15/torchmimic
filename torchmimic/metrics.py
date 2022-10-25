import numpy as np

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    mean_absolute_error,
    precision_recall_curve,
    auc,
    cohen_kappa_score,
)


def kappa(true, pred):
    """
    Returns the Cohen's Kappa for the provided true and predicted values

    :param true: true values
    :type true: np.array
    :param pred: predicted values
    :type pred: np.array
    :return: Cohen's Kappa score
    :rtype: int
    """
    pred = np.argmax(pred, axis=1)
    true = true[:, 0]
    return cohen_kappa_score(true, pred, weights="linear")


def accuracy(true, pred):
    """
    Returns the accuracy for the provided true and predicted values

    :param true: true values
    :type true: np.array
    :param pred: predicted values
    :type pred: np.array
    :return: accuracy score
    :rtype: int
    """
    return (sum(true == pred) / len(pred)) * 100


def f1(true, pred):
    """
    Returns the F1-score for the provided true and predicted values

    :param true: true values
    :type true: np.array
    :param pred: predicted values
    :type pred: np.array
    :return: F1-score
    :rtype: int
    """
    return f1_score(true, pred)


def balanced_accuracy(true, pred):
    """
    Returns the Balanced Accuracy for the provided true and predicted values

    :param true: true values
    :type true: np.array
    :param pred: predicted values
    :type pred: np.array
    :return: Balanced Accuracy score
    :rtype: int
    """
    return balanced_accuracy_score(true, pred)


def mae(true, pred):
    """
    Returns the Mean Absolute Error/Deviation for the provided
    true and predicted values

    :param true: true values
    :type true: np.array
    :param pred: predicted values
    :type pred: np.array
    :return: MAE/MAD score
    :rtype: int
    """
    one_hot = np.zeros((true.size, true.max() + 1))
    for i in np.arange(true.size):
        one_hot[np.arange(true.size), true[i]] = 1
    return mean_absolute_error(one_hot, pred)


def aucpr(true, pred):
    """
    Returns the AUC-PR for the provided true and predicted values

    :param true: true values
    :type true: np.array
    :param pred: predicted values
    :type pred: np.array
    :return: AUC-PR score
    :rtype: int
    """
    (precisions, recalls, _) = precision_recall_curve(true, pred)
    return auc(recalls, precisions)


class AUCROC:
    """
    AUCROC scoring class
    """

    def __init__(self, average=None):
        """
        Initialization for AUCROC class

        :param average: type of average used for multiclass.
        :type average: str
        """
        self.average = average

    def __call__(self, true, pred):
        """
        Returns the AUC-ROC for the provided true and predicted values

        :param true: true values
        :type true: np.array
        :param pred: predicted values
        :type pred: np.array
        :return: AUC-ROC score
        :rtype: int
        """
        return roc_auc_score(
            true, pred, multi_class="ovr", average=self.average
        )


class AverageMeter:
    """
    Class used to collect values and return a running average
    """

    def __init__(self):
        """
        Initializae the AverageMeter class
        """
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def reset(self):
        """
        Resets private members
        """
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, _n=1):
        """
        Updates class members

        :param val: value used to update running average
        :type val: float
        :param _n: sample size used to calculate value
        :type _n: int
        """
        self.sum += val * _n
        self.cnt += _n
        self.avg = self.sum / self.cnt


class MetricMeter:
    """
    Class used to collect values and evaluate them using a scoring function
    """

    def __init__(self, score_fn):
        """
        Initializae the MetricMeter Class

        :param score_fn: scoring function
        :type score_fn: function
        """
        super().__init__()
        self.reset()
        self.score_fn = score_fn

    def reset(self):
        """
        Resets private members
        """
        self.true = []
        self.pred = []

    def update(self, true, pred):
        """
        Updates list of true and predicted values

        :param true: true labels
        :type true: np.array
        :param pred: predicted labels
        :type pred: np.array
        """
        self.pred.append(pred)
        self.true.append(true)

    def score(self):
        """
        Scores true and predicted values
        :returns: the output of the score function given the predicted and true labels
        :rtype: int
        """
        self.true = np.concatenate(self.true, axis=0)
        self.pred = np.concatenate(self.pred, axis=0)
        return self.score_fn(self.true, self.pred)
