import wandb
import glob
import os

import numpy as np

from torchmimic.utils import create_exp_dir
from torchmimic.metrics import MetricMeter, AverageMeter, AUCROC, aucpr


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
            "AUC-ROC": MetricMeter(AUCROC("micro")),
            "AUC-PR": MetricMeter(aucpr),
        }

    def reset(self):
        for item in self.metrics.values():
            item.reset()

    def update(self, outputs, labels, loss):
        batch_size = outputs.size(0)

        label_tmp = labels.cpu().numpy()
        outputs = outputs.cpu().detach().numpy()

        self.metrics["Loss"].update(loss.item(), batch_size)
        self.metrics["AUC-ROC"].update(label_tmp, outputs)
        self.metrics["AUC-PR"].update(label_tmp, outputs)

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
