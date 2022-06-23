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
    def __init__(
        self,
        root,
        train=True,
        partition=10,
        steps=None,
    ):
        listfile = "train_listfile.csv" if train else "val_listfile.csv"

        self.partition = partition

        self._read_data(root, listfile)
        self._load_data(steps)

        self.steps = len(self.data)

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
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
        self.normalizer.load_params(normalizer_state)
