from __future__ import absolute_import
from __future__ import print_function

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import threading
import random
import os


class BatchGen(object):
    def __init__(
        self,
        reader,
        discretizer,
        normalizer,
        batch_size,
        small_part,
        target_repl,
        shuffle,
        return_names=False,
    ):
        self.batch_size = batch_size
        self.target_repl = target_repl
        self.shuffle = shuffle
        self.return_names = return_names

        self._load_data(reader, discretizer, normalizer, small_part)

        self.steps = (len(self.data) + batch_size - 1) // batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()

    def _load_data(self, reader, discretizer, normalizer, small_part=False):
        N = reader.get_number_of_examples()
        if small_part:
            N = 1000
        ret = read_chunk(reader, N)
        data = ret["X"]
        ts = ret["t"]
        ys = ret["y"]
        names = ret["name"]
        data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
        if normalizer is not None:
            data = [torch.Tensor(normalizer.transform(X)) for X in data]
        ys = torch.FloatTensor(ys)
        self.data = data
        self.labels = ys
        self.ts = ts
        self.names = names
        self.seq_lens = [len(d) for d in data]

    def _generator(self):
        B = self.batch_size
        while True:
            if self.shuffle:
                N = len(self.labels)
                order = list(range(N))
                random.shuffle(order)
                tmp_data = [None] * N
                tmp_sl = [None] * N
                for i in range(N):
                    tmp_data[i] = self.data[order[i]]
                    tmp_sl[i] = self.seq_lens[order[i]]
                self.data = tmp_data
                self.labels = self.labels[order]
                self.seq_lens = tmp_sl

            for i in range(0, len(self.data), B):
                x = self.data[i : i + B]
                y = self.labels[i : i + B]
                sl = self.seq_lens[i : i + B]

                x = pad_sequence(x, batch_first=True)

                batch_data = (x, y, sl)

                yield batch_data

    def __len__(self):
        return self.steps

    def __iter__(self):
        return self.generator

    def next(self):
        with self.lock:
            return next(self.generator)

    def __next__(self):
        return self.next()


def save_results(names, ts, predictions, labels, path):
    n_tasks = 25
    common_utils.create_directory(os.path.dirname(path))
    with open(path, "w") as f:
        header = ["stay", "period_length"]
        header += ["pred_{}".format(x) for x in range(1, n_tasks + 1)]
        header += ["label_{}".format(x) for x in range(1, n_tasks + 1)]
        header = ",".join(header)
        f.write(header + "\n")
        for name, t, pred, y in zip(names, ts, predictions, labels):
            line = [name]
            line += ["{:.6f}".format(t)]
            line += ["{:.6f}".format(a) for a in pred]
            line += [str(a) for a in y]
            line = ",".join(line)
            f.write(line + "\n")


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


def pad_zeros(arr, min_length=None):
    """
    `arr` is an array of `np.array`s
    The function appends zeros to every `np.array` in `arr`
    to equalize their first axis lenghts.
    """
    dtype = arr[0].dtype
    max_len = max([x.shape[0] for x in arr])
    ret = [
        np.concatenate(
            [x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0
        )
        for x in arr
    ]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [
            np.concatenate(
                [x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)],
                axis=0,
            )
            for x in ret
        ]
    return np.array(ret)


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
