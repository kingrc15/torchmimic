import os
import torch
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader

from .utils import Logger, pad_colalte
from .batch_gen import BatchGen

from ..readers import MultitaskReader
from ..preprocessing import Discretizer, Normalizer


class Multitask_Trainer:
    def __init__(
        self,
        model,
        ihm_C=1,
        los_C=1,
        pheno_C=1,
        decomp_C=1,
        train_batch_size=8,
        test_batch_size=256,
        data="/data/datasets/mimic3-benchmarks/data/multitask",
        learning_rate=0.001,
        weight_decay=0,
        report_freq=200,
        exp_name="Test",
        device="cpu",
        small_part=False,
        partition=10,
        workers=5,
    ):
        super().__init__()

        config = {
            "exp_name": exp_name,
        }

        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = model
        self.device = device
        self.report_freq = report_freq

        config.update(model.get_config())
        config.update(self.get_config())

        self.logger = Logger(config)

        torch.cuda.set_device(self.device)

        train_reader = MultitaskReader(
            dataset_dir=os.path.join(data, "train"),
            listfile=os.path.join(data, "train_listfile.csv"),
        )
        val_reader = MultitaskReader(
            dataset_dir=os.path.join(data, "train"),
            listfile=os.path.join(data, "val_listfile.csv"),
        )

        discretizer = Discretizer(
            timestep=1.0,
            store_masks=True,
            impute_strategy="previous",
            start_time="zero",
        )

        discretizer_header = discretizer.transform(
            train_reader.read_example(0)["X"]
        )[1].split(",")
        cont_channels = [
            i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1
        ]

        normalizer = Normalizer(fields=cont_channels)
        normalizer_state = "../normalizers/mult_ts1.0.input_str:previous.start_time:zero.normalizer"
        normalizer_state = os.path.join(
            os.path.dirname(__file__), normalizer_state
        )
        normalizer.load_params(normalizer_state)

        train_data_gen = BatchGen(
            train_reader,
            discretizer,
            normalizer,
            train_batch_size,
            small_part,
            small_part,
            10,
        )

        test_data_gen = BatchGen(
            val_reader,
            discretizer,
            normalizer,
            test_batch_size,
            small_part,
            small_part,
            10,
        )

        kwargs = (
            {"num_workers": workers, "pin_memory": True} if self.device else {}
        )

        self.train_loader = DataLoader(
            train_data_gen,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=pad_colalte,
            **kwargs,
        )
        self.test_loader = DataLoader(
            test_data_gen,
            batch_size=test_batch_size,
            shuffle=False,
            collate_fn=pad_colalte,
            **kwargs,
        )

        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
        )

        self.crit = nn.BCELoss()

    def fit(self, epochs):

        for epoch in range(epochs):
            self.model.train()
            self.logger.reset()
            for batch_idx, (data, label, lens, mask) in enumerate(
                self.train_loader
            ):
                data = data.to(self.device)
                label = label.to(self.device)

                output = self.model((data, lens))
                loss = self.crit(output, label)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.logger.update(output, label, loss)

                if (batch_idx + 1) % self.report_freq == 0:
                    print(
                        f"Train: epoch: {epoch+1}, loss = {self.logger.get_loss()}"
                    )

            self.logger.print_metrics(epoch, split="Train")

            self.model.eval()
            self.logger.reset()
            with torch.no_grad():
                for batch_idx, (data, label, lens, mask) in enumerate(
                    self.test_loader
                ):
                    data = data.to(self.device)
                    label = label.to(self.device)

                    output = self.model((data, lens))
                    loss = self.crit(output, label)

                    self.logger.update(output, label, loss)

                    if (batch_idx + 1) % self.report_freq == 0:
                        print(
                            f"Eval: epoch: {epoch+1}, loss = {self.logger.get_loss()}"
                        )

                self.logger.print_metrics(epoch, split="Eval")

    def get_config(self):
        return {
            "test_batch_size": self.test_batch_size,
            "train_batch_size": self.train_batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }
