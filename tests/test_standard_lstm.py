import unittest
from torchmimic.phenotyping import Phenotype_Trainer
from torchmimic.mortality import Mortality_Trainer
from torchmimic.los import LOS_Trainer
from torchmimic.decompensation import Decompensation_Trainer
from torchmimic.models.LSTM import LSTM_Model
from torchmimic.utils import get_free_gpu


class TestLSTM(unittest.TestCase):
    def test_standard_lstm_phenotype(self):
        device = get_free_gpu()

        model = LSTM_Model(
            n_classes=25,
            hidden_dim=256,
            num_layers=1,
            dropout_rate=0.3,
            bidirectional=True,
        )

        trainer = Phenotype_Trainer(
            model=model,
            train_batch_size=8,
            test_batch_size=256,
            data="/data/datasets/mimic3-benchmarks/data/phenotyping",
            learning_rate=0.001,
            weight_decay=0,
            report_freq=200,
            device=device,
            small_part=True,
        )

        trainer.fit(2)

    def test_standard_lstm_mortality(self):
        device = get_free_gpu()

        model = LSTM_Model(
            n_classes=1,
            hidden_dim=256,
            num_layers=1,
            dropout_rate=0.3,
            bidirectional=True,
        )

        trainer = Mortality_Trainer(
            model=model,
            train_batch_size=8,
            test_batch_size=256,
            data="/data/datasets/mimic3-benchmarks/data/in-hospital-mortality",
            learning_rate=0.001,
            weight_decay=0,
            report_freq=200,
            device=device,
            small_part=True,
        )

        trainer.fit(2)

    def test_standard_lstm_los(self):
        device = get_free_gpu()

        model = LSTM_Model(
            n_classes=10,
            hidden_dim=256,
            num_layers=1,
            dropout_rate=0.3,
            bidirectional=True,
        )

        trainer = LOS_Trainer(
            model=model,
            train_batch_size=8,
            test_batch_size=256,
            data="/data/datasets/mimic3-benchmarks/data/length-of-stay",
            learning_rate=0.001,
            weight_decay=0,
            report_freq=200,
            device=device,
            small_part=True,
            partition=10,
        )

        trainer.fit(2)

    def test_standard_lstm_decomp(self):
        device = get_free_gpu()

        model = LSTM_Model(
            n_classes=1,
            hidden_dim=256,
            num_layers=1,
            dropout_rate=0.3,
            bidirectional=True,
        )

        trainer = Decompensation_Trainer(
            model=model,
            train_batch_size=8,
            test_batch_size=256,
            data="/data/datasets/mimic3-benchmarks/data/decompensation",
            learning_rate=0.001,
            weight_decay=0,
            report_freq=200,
            device=device,
            small_part=True,
            partition=10,
        )

        trainer.fit(2)
