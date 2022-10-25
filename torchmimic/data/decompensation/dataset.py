from __future__ import absolute_import
from __future__ import print_function

import random
import os
import torch

import numpy as np

from torchmimic.data.preprocessing import Discretizer, Normalizer
from torchmimic.data.readers import DecompensationReader
from torchmimic.data.utils import read_chunk
from torchmimic.data.base_dataset import BaseDataset


class DecompensationDataset(BaseDataset):
    """
    Decompensation dataset that can be directly used by PyTorch dataloaders. This class preprocessing the data the same way as "Multitask learning and benchmarking with clinical time series data": https://github.com/YerevaNN/mimic3-benchmarks

    :param root: directory where data is located
    :type root: str
    :param train: if true, the training split of the data will be used. Otherwise, the validation dataset will be used
    :type train: bool
    :param n_samples: number of samples to use. If None, all the data is used
    :type steps: int
    """

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        n_samples=None,
    ):
        """
        Initialize DecompensationDataset

        :param root: directory where data is located
        :type root: str
        :param train: if true, the training split of the data will be used. Otherwise, the validation dataset will be used
        :type train: bool
        :param n_samples: number of samples to use. If None, all the data is used
        :type steps: int
        """
        super().__init__(transform=transform)

        listfile = "train_listfile.csv" if train else "val_listfile.csv"

        self._read_data(root, listfile)
        self._load_data(n_samples)

        self.n_samples = len(self.data)

    def _read_data(self, root, listfile):
        self.reader = DecompensationReader(
            dataset_dir=os.path.join(root, "train"),
            listfile=os.path.join(root, listfile),
        )

        self.discretizer = Discretizer(
            timestep=1.0,
            store_masks=True,
            impute_strategy="previous",
            start_time="zero",
        )

        discretizer_header = self.discretizer.transform(
            self.reader.read_example(0)["X"]
        )[1].split(",")
        cont_channels = [
            i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1
        ]

        self.normalizer = Normalizer(fields=cont_channels)
        normalizer_state = "../normalizers/decomp_ts1.0.input_str:previous.n1e5.start_time:zero.normalizer"
        normalizer_state = os.path.join(
            os.path.dirname(__file__), normalizer_state
        )
        self.normalizer.load_params(normalizer_state)
