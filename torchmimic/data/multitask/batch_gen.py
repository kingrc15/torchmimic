from __future__ import absolute_import
from __future__ import print_function

import torch

import numpy as np

import random
import os

from ..utils import get_bin_custom, read_chunk


class BatchGen(object):
    def __init__(
        self,
        reader,
        discretizer,
        normalizer,
        batch_size,
        small_part,
        shuffle,
        partition,
        return_mask=False,
    ):
        self.discretizer = discretizer
        self.normalizer = normalizer
        self.ihm_pos = int(48.0 / 1 - 1e-6)
        self.partition = partition
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._load_data(reader, discretizer, normalizer, small_part)

        self.steps = len(self.data)

    def _load_data(self, reader, discretizer, normalizer, small_part=False):
        N = reader.get_number_of_examples()
        if small_part:
            N = 1000

        ret = read_chunk(reader, N)
        Xs = ret["X"]
        ts = ret["t"]
        ihms = ret["ihm"]
        loss = ret["los"]
        phenos = ret["pheno"]
        decomps = ret["decomp"]

        self.data = dict()
        self.data["pheno_ts"] = ts
        self.data["names"] = ret["name"]
        self.data["decomp_ts"] = []
        self.data["los_ts"] = []

        for i in range(N):
            self.data["decomp_ts"].append(
                [pos for pos, m in enumerate(decomps[i][0]) if m == 1]
            )
            self.data["los_ts"].append(
                [pos for pos, m in enumerate(loss[i][0]) if m == 1]
            )
            (
                Xs[i],
                ihms[i],
                decomps[i],
                loss[i],
                phenos[i],
            ) = self._preprocess_single(
                Xs[i], ts[i], ihms[i], decomps[i], loss[i], phenos[i]
            )

        self.data["X"] = Xs
        self.data["ihm_M"] = [x[0] for x in ihms]
        self.data["ihm_y"] = [x[1] for x in ihms]
        self.data["decomp_M"] = [x[0] for x in decomps]
        self.data["decomp_y"] = [x[1] for x in decomps]
        self.data["los_M"] = [x[0] for x in loss]
        self.data["los_y"] = [x[1] for x in loss]
        self.data["pheno_y"] = phenos

        self.seq_lens = [len(Xs[0]) for x in ihms]

    def _preprocess_single(self, X, max_time, ihm, decomp, los, pheno):
        timestep = self.discretizer._timestep
        eps = 1e-6

        def get_bin(t):
            return int(t / timestep - eps)

        n_steps = get_bin(max_time) + 1

        # X
        X = self.discretizer.transform(X, end=max_time)[0]
        if self.normalizer is not None:
            X = self.normalizer.transform(X)
        assert len(X) == n_steps

        # ihm
        # NOTE: when mask is 0, we set y to be 0. This is important
        #       because in the multitask networks when ihm_M = 0 we set
        #       our prediction thus the loss will be 0.
        if np.equal(ihm[1], 0):
            ihm[2] = 0
        ihm = (np.int32(ihm[1]), np.int32(ihm[2]))  # mask, label

        # decomp
        decomp_M = [0] * n_steps
        decomp_y = [0] * n_steps
        for i in range(len(decomp[0])):
            pos = get_bin(i)
            decomp_M[pos] = decomp[0][i]
            decomp_y[pos] = decomp[1][i]
        decomp = (
            np.array(decomp_M, dtype=np.int32),
            np.array(decomp_y, dtype=np.int32),
        )

        # los
        los_M = [0] * n_steps
        los_y = [0] * n_steps
        for i in range(len(los[0])):
            pos = get_bin(i)
            los_M[pos] = los[0][i]
            los_y[pos] = los[1][i]
        los = (
            np.array(los_M, dtype=np.int32),
            np.array(los_y, dtype=np.float32),
        )

        # pheno
        pheno = np.array(pheno, dtype=np.int32)

        return (X, ihm, decomp, los, pheno)

    def get_max_seq_length(self):
        return max(self.seq_lens)

    def __getitem__(self, idx):
        outputs = []

        # X
        X = self.data["X"][idx]
        T = X.shape[1]

        ihm_M = np.array(self.data["ihm_M"][idx])
        ihm_M = np.expand_dims(ihm_M, axis=-1)  # (B, 1)
        ihm_y = np.array(self.data["ihm_y"][idx])
        ihm_y = torch.FloatTensor()  # (B, 1)
        outputs.append(ihm_y)

        # decomp
        decomp_M = self.data["decomp_M"][idx]
        decomp_y = self.data["decomp_y"][idx]
        decomp_y = torch.FloatTensor(
            np.expand_dims(decomp_y, axis=-1)
        )  # (B, T, 1)
        outputs.append(decomp_y)

        # los
        los_M = self.data["los_M"][idx]
        los_y = self.data["los_y"][idx]

        los_y = np.array([get_bin_custom(ly, 10) for ly in los_y], dtype=int)
        los_y = torch.FloatTensor(los_y)
        outputs.append(los_y)

        # pheno
        pheno_y = torch.FloatTensor(self.data["pheno_y"][idx])
        outputs.append(pheno_y)

        inputs = [X, ihm_M, decomp_M, los_M]

        return inputs, outputs, self.seq_lens[idx]

    def __len__(self):
        return self.steps
