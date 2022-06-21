from .phenotyping import Phenotype_Trainer
from .models.LSTM import LSTM_Model
from .utils import get_free_gpu


def standard_lstm_benchmark():
    device = get_free_gpu()

    model = LSTM_Model(
        n_classes=25, hidden_dim=256, num_layers=1, dropout_rate=0.3, bidirectional=True
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
    )

    trainer.fit(100)


if __name__ == "__main__":
    standard_lstm_benchmark()
