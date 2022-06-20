import os
import torch
import shutil
import threading
import random
import wandb
import glob

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import pearsonr

from sklearn.manifold import TSNE
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    mean_absolute_error,
)

import warnings

warnings.filterwarnings("ignore")


def get_free_gpu():
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    return int(np.argmax(memory_available))


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
        mean = self.mean.to(x.device).expand_as(x)
        std = self.std.to(x.device).expand_as(x)
        return x.sub_(mean).div_(std)

    def inverse(self, x):
        mean = self.mean.to(x.device).expand_as(x)
        std = self.std.to(x.device).expand_as(x)
        return x.mul_(std).add_(mean)


class Preprocess:
    def __init__(self, steps):
        self.steps = steps

    def __call__(self, x):
        for step in self.steps:
            x = step(x)

        return x


def frame_count(frame_size, window_size, stride):
    return max(int(((frame_size - (window_size - 1) - 1) / stride) + 1), 0)


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
        plt.title(f"Reference vs Estimate. Correlation = {pearsonr(Y, Y_pred)[0]:.5f}")
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


class ROCAUC:
    def __init__(self, type="micro"):
        self.type = type

    def __call__(self, Y, Y_pred):
        return roc_auc_score(Y, Y_pred, multi_class="ovr", average=self.type)


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6


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

class Logger:
    def __init__(self, config):
        wandb.init(project="MIMIC Benchmark", name=config.exp_name)
        wandb.config.update(config)

        self.experiment_path = f"./exp/{config.exp_name}"

        wandb.run.log_code("*.py")
        create_exp_dir(self.experiment_path, scripts_to_save=glob.glob("*.py"))
        np.save(os.path.join(self.experiment_path, "config"), config)

        self.metrics = {
            "Loss": AverageMeter(),
            "AUC-ROC Micro": MetricMeter(ROCAUC("micro")),
            "AUC-ROC Macro": MetricMeter(ROCAUC("macro"))
        }

    def reset(self):
        for item in self.metrics.values():
            item.reset()

    def update(self, outputs, labels, loss):
        batch_size = outputs.size(0)
        
        label_tmp = labels.cpu().numpy()
        outputs = outputs.cpu().detach().numpy()

        self.metrics["Loss"].update(loss.item(), batch_size)
        self.metrics["AUC-ROC Micro"].update(label_tmp, outputs)
        self.metrics["AUC-ROC Macro"].update(label_tmp, outputs)

    def get_loss(self):
        return self.metrics["Loss"].avg

    def print_metrics(self, epoch, split="Train"):
        assert split=="Train" or split=="Eval"

        result_str = split + ": "

        wandb.log({"Epoch": epoch + 1}, commit=False)
        result_str += f" Epoch {epoch+1}"
        for name, meter in self.metrics.items():
            if isinstance(meter, MetricMeter):
                result = meter.score()
            elif isinstance(meter, AverageMeter):
                result = meter.avg

            wandb.log({split + " " + name: result}, commit=False)
            result_str += f", {name}={result}"

        print(result_str)
        wandb.log({}, commit=True)

    def save(self, model):
        torch.save(model.state_dict(), os.path.join(self.experiment_path, "weights.pt"))            
