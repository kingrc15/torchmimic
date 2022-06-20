import argparse


def parse():
    parser = argparse.ArgumentParser("uq-quant")

    # Data
    parser.add_argument(
        "--data",
        type=str,
        default="/data/datasets/mimic3-benchmarks/data/phenotyping",
        help="location of the data corpus",
    )
    parser.add_argument("--train_batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--test_batch_size", type=int, default=256, help="batch size")

    # Training
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="init learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument(
        "--report_freq", type=float, default=200, help="report frequency"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="num of training epochs"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=30,
        help="sequence length used by sliding window",
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="stride used by sliding window"
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.4, help="residual dropout rate"
    )
    parser.add_argument(
        "--exp_name", type=str, default="test", help="experiment name"
    )

    # Results
    parser.add_argument(
        "--dst_folder", type=str, default="exp", help="destination folder for results"
    )

    # Model
    parser.add_argument(
        "--model_type",
        choices=["LSTM", "SAnD"],
        help="regression or classification",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        default=False,
        help="use bidirectional recurrent layers",
    )
    parser.add_argument("--hidden_dim", type=int, default=256, help="hidden layer size")
    parser.add_argument("--num_layers", type=int, default=1, help="number of layers")

    # Hardware
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")

    return parser.parse_args()
