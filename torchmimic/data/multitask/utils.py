import wandb
import glob
import os

import numpy as np

from torch.nn.utils.rnn import pad_sequence

from ..utils import create_exp_dir
from ..metrics import MetricMeter, AverageMeter, AUCROC


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
            "AUC-ROC Micro": MetricMeter(AUCROC("micro")),
            "AUC-ROC Macro": MetricMeter(AUCROC("macro")),
        }

    def update(self, outputs, labels, loss):
        batch_size = outputs.size(0)

        label_tmp = labels.cpu().numpy()
        outputs = outputs.cpu().detach().numpy()

        self.metrics["Loss"].update(loss.item(), batch_size)
        self.metrics["AUC-ROC Micro"].update(label_tmp, outputs)
        self.metrics["AUC-ROC Macro"].update(label_tmp, outputs)
