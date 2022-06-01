import argparse


def parse():
    parser = argparse.ArgumentParser("uq-quant")

    # Data
    parser.add_argument(
        "--data", type=str, default="../data", help="location of the data corpus"
    )
    parser.add_argument("--train_batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--test_batch_size", type=int, default=1024, help="batch size")

    # Training
    parser.add_argument(
        "--learning_rate", type=float, default=0.003, help="init learning rate"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="weight decay"
    )
    parser.add_argument(
        "--report_freq", type=float, default=200, help="report frequency"
    )
    parser.add_argument(
        "--epochs", type=int, default=500, help="num of training epochs"
    )
    parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")

    # Results
    parser.add_argument(
        "--dst_folder", type=str, default="exp", help="destination folder for results"
    )
    
    # Model
    parser.add_argument(
        "--model_type", 
        choices=["LSTM"],
        help="regression or classification",
    )

    # Hardware
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")

    return parser.parse_args()
