import glob
import os

from abc import ABC, abstractmethod

import numpy as np
import torch
import wandb

from torchmimic.metrics import AverageMeter, MetricMeter
from torchmimic.utils import create_exp_dir


class BaseLogger(ABC):
    """
    Base Logger class. Used for logging, printing, and saving information about the run. Contains built-in wandb support.

    :param config: A dictionary of the run configuration
    :type config: dict
    :param log_wandb: If true, wandb will be used to log metrics and configuration
    :type log_wandb: bool
    """

    def __init__(self, exp_name, config, log_wandb=False):
        """
        Initialize BaseLogger

        :param config: A dictionary of the run configuration
        :type config: dict
        :param log_wandb: If true, wandb will be used to log metrics and configuration
        :type log_wandb: bool
        """
        self.log_wandb = log_wandb

        if self.log_wandb:
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
        }

    def __del__(self):
        """
        Destructor for BaseLogger. Finishes wandb if log_wandb is true
        """
        if self.log_wandb:
            wandb.finish()

    @abstractmethod
    def update(self, outputs, labels, loss):
        """
        Abstract class for updating metrics
        """
        pass

    def reset(self):
        """
        Resets metrics
        """
        for item in self.metrics.values():
            item.reset()

    def get_loss(self):
        """
        Returns average loss

        :return: Average Loss
        :rtype: float
        """
        return self.metrics["Loss"].avg

    def print_metrics(self, epoch, split="Train"):
        """
        Prints and logs metrics. If log_wandb is True, wandb run will be updated

        :param epoch: The current epoch
        :type epoch: int
        :param split: The split of the data. "Train" or "Eval"
        :type split: str
        """

        assert split in ("Train", "Eval")

        result_str = split + ": "

        if self.log_wandb:
            wandb.log({"Epoch": epoch + 1}, commit=False)

        result_str += f" Epoch {epoch+1}"
        for name, meter in self.metrics.items():
            if isinstance(meter, MetricMeter):
                result = meter.score()
            elif isinstance(meter, AverageMeter):
                result = meter.avg

            if self.log_wandb:
                wandb.log({split + " " + name: result}, commit=False)
            result_str += f", {name}={result}"

        print(result_str)
        if split == "Eval" and self.log_wandb:
            wandb.log({})

    def save(self, model):
        """
        Saves the provides models to the experiment path
        """
        torch.save(
            model.state_dict(),
            os.path.join(self.experiment_path, "weights.pt"),
        )
