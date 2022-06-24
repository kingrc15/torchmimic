from __future__ import absolute_import
from __future__ import print_function

import random
import os
import torch

import numpy as np

from abc import ABC, abstractmethod
from torchmimic.data.preprocessing import Discretizer, Normalizer
from torchmimic.data.utils import read_chunk


class BaseDataset(ABC):
    @abstractmethod
    def _read_data(self, root, listfile):
        pass

    def _load_data(self, sample_size):
        N = self.reader.get_number_of_examples()

        if sample_size is None:
            sample_size = N

        ret = read_chunk(self.reader, sample_size)

        data = ret["X"]
        ts = ret["t"]
        ys = ret["y"]
        names = ret["name"]

        data = [self.discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
        if self.normalizer is not None:
            data = [torch.Tensor(self.normalizer.transform(X)) for X in data]
        ys = torch.FloatTensor(ys)
        self.data = data
        self.labels = ys
        self.ts = ts
        self.names = names
        self.seq_lens = [len(d) for d in data]

    def __getitem__(self, idx):
        x = self.data[idx]
        sl = self.seq_lens[idx]
        y = self.labels[idx]

        return x, y, sl

    def __len__(self):
        return self.n_samples
