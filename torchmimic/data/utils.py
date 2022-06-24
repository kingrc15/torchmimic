import torch


def read_chunk(reader, chunk_size):
    data = {}
    for _ in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]
    return data


class CustomBins:
    inf = 1e18
    bins = [
        (-inf, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 14),
        (14, +inf),
    ]
    nbins = len(bins)
    means = [
        11.450379,
        35.070846,
        59.206531,
        83.382723,
        107.487817,
        131.579534,
        155.643957,
        179.660558,
        254.306624,
        585.325890,
    ]


def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0] * 24.0
        b = CustomBins.bins[i][1] * 24.0
        if a <= x < b:
            if one_hot:
                ret = torch.zeros((CustomBins.nbins,))
                ret[i] = 1
                return int(ret)
            return torch.Tensor([i]).long()
    return None
