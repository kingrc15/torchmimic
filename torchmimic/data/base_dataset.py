from __future__ import absolute_import
from __future__ import print_function

from abc import ABC, abstractmethod

import torch
import numpy

from torchmimic.data.utils import read_chunk


class BaseDataset(ABC):
    def __init__(self, transform):
        self.transform = transform
        self.reader = None
        self.discretizer = None
        self.normalizer = None

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

        data_tmp = []
        self.mask = []

        for X, t in zip(data, ts):
            d = self.discretizer.transform(X, end=t)[0]
            data_tmp.append(d)
            self.mask.append(self.expand_mask(d[:, 59:]))

        # data = [self.discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
        if self.normalizer is not None:
            self.data = [
                torch.Tensor(self.normalizer.transform(X)) for X in data_tmp
            ]
        ys = torch.FloatTensor(ys)
        self.labels = ys
        self.ts = ts
        self.names = names

    def __getitem__(self, idx):
        x = self.data[idx]
        sl = len(x)
        y = self.labels[idx]
        m = self.mask[idx]

        if self.transform:
            x = self.transform(x)

        return x, y, sl, m

    def __len__(self):
        return self.n_samples

    def expand_mask(self, mask):
        expanded_mask = torch.ones((mask.shape[0], 59))

        for i, pv in enumerate(self.discretizer._possible_values.values()):
            n_values = len(pv) if not pv == [] else 1
            for p in range(n_values):
                expanded_mask[:, p + i] = torch.from_numpy(mask[:, i])

        return expanded_mask
