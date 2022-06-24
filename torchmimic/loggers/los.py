from torchmimic.metrics import MetricMeter, kappa, mae
from .base_logger import BaseLogger


class LOSLogger(BaseLogger):
    """
    Length-of-Stay Logger class. Used for logging, printing, and saving information about the run. Logs loss, Cohen's Kappa and Mean Absolute Deviation. Contains built-in wandb support.

    :param config: A dictionary of the run configuration
    :type config: dict
    :param log_wandb: If true, wandb will be used to log metrics and configuration
    :type log_wandb: bool
    """

    def __init__(self, exp_name, config, log_wandb=False):
        """
        Initialize LOSLogger

        :param config: A dictionary of the run configuration
        :type config: dict
        :param log_wandb: If true, wandb will be used to log metrics and configuration
        :type log_wandb: bool
        """
        super().__init__(exp_name, config, log_wandb=log_wandb)

        self.metrics.update(
            {
                "Cohen Kappa": MetricMeter(kappa),
                "MAD": MetricMeter(mae),
            }
        )

    def update(self, outputs, labels, loss):
        """
        Update loss, Cohen's Kappa and Mean Absolute Deviation

        :param outputs: Predicted labels
        :type outputs: torch.Tensor
        :param labels: True labels
        :type labels: torch.Tensor
        :param loss: Loss from the training iteration.
        :type loss: float
        """

        batch_size = outputs.size(0)

        label_tmp = labels.cpu().numpy()
        outputs = outputs.cpu().detach().numpy()

        self.metrics["Loss"].update(loss.item(), batch_size)
        self.metrics["Cohen Kappa"].update(label_tmp, outputs)
        self.metrics["MAD"].update(label_tmp, outputs)
