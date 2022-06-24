from torchmimic.metrics import MetricMeter, AUCROC
from .base_logger import BaseLogger


class PhenotypingLogger(BaseLogger):
    """
    Phenotyping Logger class. Used for logging, printing, and saving information about the run. Logs loss, AUC-ROC macro, and AUC-ROC micro. Contains built-in wandb support.

    :param config: A dictionary of the run configuration
    :type config: dict
    :param log_wandb: If true, wandb will be used to log metrics and configuration
    :type log_wandb: bool
    """

    def __init__(self, exp_name, config, log_wandb=False):
        """
        Initialize PhenotypingLogger

        :param config: A dictionary of the run configuration
        :type config: dict
        :param log_wandb: If true, wandb will be used to log metrics and configuration
        :type log_wandb: bool
        """

        super().__init__(exp_name, config, log_wandb=log_wandb)

        self.metrics.update(
            {
                "AUC-ROC Micro": MetricMeter(AUCROC("micro")),
                "AUC-ROC Macro": MetricMeter(AUCROC("macro")),
            }
        )

    def update(self, outputs, labels, loss):
        """
        Update loss, AUC-ROC macro, and AUC-ROC micro

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
        self.metrics["AUC-ROC Micro"].update(label_tmp, outputs)
        self.metrics["AUC-ROC Macro"].update(label_tmp, outputs)
