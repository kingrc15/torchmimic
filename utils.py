import os
import torch
import shutil
import threading
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import pearsonr

from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, mean_absolute_error

import warnings

warnings.filterwarnings("ignore")


def remove_nans(x):
    return x[~np.isnan(x).any(axis=1)]


def test_range(x):
    if len(x) > 0:
        maxs = x.max(0)
        if maxs[2] > 80:
            return x

    return []


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

    def __call__(self, x):
        mean = self.mean.to(x.device).view(-1, 1)
        std = self.std.to(x.device).view(-1, 1)
        return x.sub_(mean).div_(std)

    def inverse(self, x):
        print(x.size(), self.mean.size())
        mean = self.mean.to(x.device).view(-1, 1)
        std = self.std.to(x.device).view(-1, 1)
        return x.mul_(std).add_(mean)


class Preprocess:
    def __init__(self, steps):
        self.steps = steps

    def __call__(self, x):
        for step in self.steps:
            x = step(x)

        return x


class BlandAltman:
    def __init__(self, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.save_path = save_path

    def __call__(self, Y, Y_pred):
        a, b = np.polyfit(Y, Y_pred, 1)

        plt.clf()
        plt.scatter(Y, Y_pred, marker=".")
        plt.axline((0, b), (1, a + b), color="k", label=f"{a}x + {b}")
        plt.title(
            f"Reference vs Estimate. Correlation = {pearsonr(Y, Y_pred)[0]:.5f}"
        )
        plt.xlabel("True Potassium")
        plt.ylabel("Estimated Potassium")
        plt.legend()
        plt.xlim([2.5, 6])
        plt.savefig(os.path.join(self.save_path, f"Bland-Altman Ref vs Est"))

        diff = Y - Y_pred
        md = np.mean(diff)
        sd = np.std(diff)
        
        print(f"Mean = {md}, Standard Deviation = {sd}")

        plt.clf()
        plt.scatter(Y, diff, marker=".")
        plt.axhline(md, color="k", linestyle="--")
        plt.axhline(md + 1.96 * sd, color="r", linestyle="--", label=f"{sd:.5f}")
        plt.axhline(md - 1.96 * sd, color="r", linestyle="--", label=f"{-1 * sd:.5f}")
        plt.title("Bland-Altman")
        plt.xlabel("True Potassium")
        plt.ylabel("Residuals")
        plt.savefig(os.path.join(self.save_path, "Bland-Altman Ref vs Res"))


def plot_latent(x, filename):
    plt.clf()
    x_proj = tsne.fit_transform(x)
    plt.scatter(x_proj[:, 0], x_proj[:, 1])
    plt.savefig(filename)


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

def rocauc(Y, Y_pred):
    try:
        return roc_auc_score(Y, Y_pred.T)
    except Exception as e:
        print(Y, Y_pred)
        print(e)
        return 0


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6


class AverageMeter:
    """docstring for AverageMeter"""

    def __init__(self):
        """AverageMeter initialization"""
        self.reset()

    def reset(self):
        """AverageMeter set initial values for class parameters"""
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, _n=1):
        """update class parameters"""
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
        self.Y = np.hstack(self.Y)
        self.Y_pred = np.hstack(self.Y_pred)
        return self.score_fn(self.Y, self.Y_pred)


def waveform_rmse(targets, predictions):
    sbp_count = 0
    dbp_count = 0
    sbp_rmse = 0
    dbp_rmse = 0

    target_ecgs = targets[:, 0]
    prediction_ecgs = predictions[:, 0]

    for prediction, target in zip(predictions, targets):
        peaks = nk.ecg_findpeaks(target[0], sampling_rate=120)["ECG_R_Peaks"]

        SBP_data = []
        DBP_data = []
        beat_2_beat_wf = []

        for i in range(len(peaks) - 1):

            SBP_data.append(np.argmax(target[2, peaks[i] : peaks[i + 1]]))
            DBP_data.append(np.argmin(target[2, peaks[i] : peaks[i + 1]]))

        sbp_count += len(SBP_data)
        dbp_count += len(DBP_data)

        for peak in SBP_data:
            sbp_rmse += (prediction[2, peak] - target[2, peak]) ** 2

        for peak in DBP_data:
            dbp_rmse += (prediction[2, peak] - target[2, peak]) ** 2

    return np.sqrt(sbp_rmse / sbp_count), np.sqrt(dbp_rmse / dbp_count)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print("Experiment dir: {}".format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, "scripts")):
            os.mkdir(os.path.join(path, "scripts"))
        for script in scripts_to_save:
            dst_file = os.path.join(path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)
            
            
class BatchGen(object):

    def __init__(self, reader, discretizer, normalizer, batch_size,
                 small_part, target_repl, shuffle, return_names=False):
        self.batch_size = batch_size
        self.target_repl = target_repl
        self.shuffle = shuffle
        self.return_names = return_names

        self._load_data(reader, discretizer, normalizer, small_part)

        self.steps = (len(self.data[0]) + batch_size - 1) // batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()

    def _load_data(self, reader, discretizer, normalizer, small_part=False):
        N = reader.get_number_of_examples()
        if small_part:
            N = 1000
        ret = read_chunk(reader, N)
        data = ret["X"]
        ts = ret["t"]
        ys = ret["y"]
        names = ret["name"]
        data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
        if (normalizer is not None):
            data = [normalizer.transform(X) for X in data]
        ys = np.array(ys, dtype=np.int32)
        self.data = (data, ys)
        self.ts = ts
        self.names = names

    def _generator(self):
        B = self.batch_size
        while True:
            if self.shuffle:
                N = len(self.data[1])
                order = list(range(N))
                random.shuffle(order)
                tmp_data = [[None] * N, [None] * N]
                tmp_names = [None] * N
                tmp_ts = [None] * N
                for i in range(N):
                    tmp_data[0][i] = self.data[0][order[i]]
                    tmp_data[1][i] = self.data[1][order[i]]
                    tmp_names[i] = self.names[order[i]]
                    tmp_ts[i] = self.ts[order[i]]
                self.data = tmp_data
                self.names = tmp_names
                self.ts = tmp_ts
            else:
                # sort entirely
                X = self.data[0]
                y = self.data[1]
                (X, y, self.names, self.ts) = common_utils.sort_and_shuffle([X, y, self.names, self.ts], B)
                self.data = [X, y]

            self.data[1] = np.array(self.data[1])  # this is important for Keras
            for i in range(0, len(self.data[0]), B):
                x = self.data[0][i:i+B]
                y = self.data[1][i:i+B]
                names = self.names[i:i + B]
                ts = self.ts[i:i + B]

                x = pad_zeros(x)
                y = np.array(y)  # (B, 25)

                if self.target_repl:
                    y_rep = np.expand_dims(y, axis=1).repeat(x.shape[1], axis=1)  # (B, T, 25)
                    batch_data = (x, [y, y_rep])
                else:
                    batch_data = (x, y)

                if not self.return_names:
                    yield batch_data
                else:
                    yield {"data": batch_data, "names": names, "ts": ts}

    def __iter__(self):
        return self.generator

    def next(self):
        with self.lock:
            return next(self.generator)

    def __next__(self):
        return self.next()
    
def read_chunk(reader, chunk_size):
    data = {}
    for i in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]
    return data

def pad_zeros(arr, min_length=None):
    """
    `arr` is an array of `np.array`s
    The function appends zeros to every `np.array` in `arr`
    to equalize their first axis lenghts.
    """
    dtype = arr[0].dtype
    max_len = max([x.shape[0] for x in arr])
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret)
