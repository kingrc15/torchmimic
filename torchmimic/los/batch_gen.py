from __future__ import absolute_import
from __future__ import print_function

import torch

import numpy as np

import random
import os

from .utils import get_bin_custom


class BatchGen(object):
    def __init__(
        self,
        reader,
        discretizer,
        normalizer,
        partition,
        steps,
        shuffle,
    ):
        self.reader = reader
        self.partition = partition
        self.discretizer = discretizer
        self.normalizer = normalizer
        self.shuffle = shuffle

        self._load_data(reader, discretizer, normalizer, steps)

        self.steps = len(self.data)

    def _load_data(self, reader, discretizer, normalizer, steps):
        N = reader.get_number_of_examples()
        ret = read_chunk(reader, steps)
        data = ret["X"][:steps]
        ts = ret["t"][:steps]
        ys = ret["y"][:steps]
        names = ret["name"][:steps]
        data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
        if normalizer is not None:
            data = [torch.Tensor(normalizer.transform(X)) for X in data]
        ys = torch.FloatTensor(ys)
        self.data = data
        self.labels = ys
        self.ts = ts
        self.names = names
        self.seq_lens = [len(d) for d in data]

    def get_max_seq_length(self):
        return max(self.seq_lens)

    def __getitem__(self, idx):
        x = self.data[idx]
        sl = self.seq_lens[idx]

        if self.partition == 10:
            y = get_bin_custom(self.labels[idx], 10)
        else:
            y = self.labels[idx]

        return x, y, sl

    def __len__(self):
        return self.steps


def read_chunk(reader, chunk_size):
    data = {}
    for i in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]
    return data


def sort_and_shuffle(data, batch_size):
    """Sort data by the length and then make batches and shuffle them.
    data is tuple (X1, X2, ..., Xn) all of them have the same length.
    Usually data = (X, y).
    """
    assert len(data) >= 2
    data = list(zip(*data))

    random.shuffle(data)

    old_size = len(data)
    rem = old_size % batch_size
    head = data[: old_size - rem]
    tail = data[old_size - rem :]
    data = []

    head.sort(key=(lambda x: x[0].shape[0]))

    mas = [head[i : i + batch_size] for i in range(0, len(head), batch_size)]
    random.shuffle(mas)

    for x in mas:
        data += x
    data += tail

    data = list(zip(*data))
    return data
