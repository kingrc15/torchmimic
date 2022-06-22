import wandb
import glob
import os
import torch

import numpy as np

from ..utils import create_exp_dir

from ..metrics import MetricMeter, AverageMeter, kappa, mae

class CustomBins:
    inf = 1e18
    bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
    nbins = len(bins)
    means = [11.450379, 35.070846, 59.206531, 83.382723, 107.487817,
             131.579534, 155.643957, 179.660558, 254.306624, 585.325890]


def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0] * 24.0
        b = CustomBins.bins[i][1] * 24.0
        if a <= x < b:
            if one_hot:
                ret = torch.zeros((CustomBins.nbins,))
                ret[i] = 1
                return int(ret)
            return torch.Tensor([i]).long()
    return None


class Logger:
    def __init__(self, config, wandb=False):
        exp_name = config["exp_name"]
        self.wandb = wandb

        if wandb:
            wandb.init(project="MIMIC Benchmark", name=exp_name)
            wandb.config.update(config)
            wandb.run.log_code("*.py")

        self.experiment_path = f"./exp/{exp_name}"

        if not os.path.exists("./exp"):
            os.mkdir("./exp")

        create_exp_dir(self.experiment_path, scripts_to_save=glob.glob("*.py"))
        np.save(os.path.join(self.experiment_path, "config"), config)

        self.metrics = {
            "Loss": AverageMeter(),
            "Cohen Kappa": MetricMeter(kappa),
            "MAD": MetricMeter(mae),
        }

    def reset(self):
        for item in self.metrics.values():
            item.reset()

    def update(self, outputs, labels, loss):
        batch_size = outputs.size(0)

        label_tmp = labels.cpu().numpy()
        outputs = outputs.cpu().detach().numpy()

        self.metrics["Loss"].update(loss.item(), batch_size)
        self.metrics["Cohen Kappa"].update(label_tmp, outputs)
        self.metrics["MAD"].update(label_tmp, outputs)

    def get_loss(self):
        return self.metrics["Loss"].avg

    def print_metrics(self, epoch, split="Train"):
        assert split == "Train" or split == "Eval"

        result_str = split + ": "

        if self.wandb:
            wandb.log({"Epoch": epoch + 1}, commit=False)

        result_str += f" Epoch {epoch+1}"
        for name, meter in self.metrics.items():
            if isinstance(meter, MetricMeter):
                result = meter.score()
            elif isinstance(meter, AverageMeter):
                result = meter.avg

            if self.wandb:
                wandb.log({split + " " + name: result}, commit=False)
            result_str += f", {name}={result}"

        print(result_str)
        if split == "Eval" and self.wandb:
            wandb.log({})

    def save(self, model):
        torch.save(model.state_dict(), os.path.join(self.experiment_path, "weights.pt"))
