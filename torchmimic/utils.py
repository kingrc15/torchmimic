import os
import torch
import shutil
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import pearsonr
from torch.nn.utils.rnn import pad_sequence

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


def pad_colalte(batch):
    xx, yy, lens = zip(*batch)
    x = pad_sequence(xx, batch_first=True, padding_value=-np.inf)
    y = torch.stack(yy, dim=0)

    mask = (x == -np.inf)[:, :, 0]
    return x, y, lens, mask


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


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6


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
