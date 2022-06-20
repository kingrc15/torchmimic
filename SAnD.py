import torch
import torch.nn as nn

import modules


class EncoderLayerForSAnD(nn.Module):
    def __init__(self, input_features, seq_len, n_heads, n_layers, d_model=128, dropout_rate=0.2) -> None:
        super(EncoderLayerForSAnD, self).__init__()
        self.d_model = d_model

        # self.input_embedding = nn.Conv1d(input_features, d_model, 3, padding=1)
        self.input_embedding = nn.Linear(input_features, d_model)
        self.positional_encoding = modules.PositionalEncoding(d_model, seq_len)
        self.time_encoding = modules.TimeEncoding(d_model, seq_len)
        self.blocks = nn.ModuleList([
            modules.Block(d_model, n_heads) for _ in range(n_layers)
        ])

    def forward(self, x, ts=None) -> torch.Tensor:
        if x.isnan().any():
            print(x)
            raise
        # x = x.transpose(1, 2)
        x = self.input_embedding(x)
        
        # x = x.transpose(1, 2)

        if ts:
            x = self.time_encoding(x, ts)
        else:
            x = self.positional_encoding(x)

        for l in self.blocks:
            x = l(x)

        return x


class SAnD(nn.Module):
    """
    Simply Attend and Diagnose model

    The Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18)

    `Attend and Diagnose: Clinical Time Series Analysis Using Attention Models <https://arxiv.org/abs/1711.03905>`_
    Huan Song, Deepta Rajan, Jayaraman J. Thiagarajan, Andreas Spanias
    """
    def __init__(
            self, input_features: int, seq_len: int, n_heads: int, factor: int,
            n_class: int, n_layers: int, d_model: int = 128, dropout_rate: float = 0.2
    ) -> None:
        super(SAnD, self).__init__()
        self.encoder = EncoderLayerForSAnD(input_features, seq_len, n_heads, n_layers, d_model, dropout_rate)
        # self.dense_interpolation = modules.DenseInterpolation(seq_len, factor)
        self.clf = modules.ClassificationModule(d_model, seq_len, n_class)
        # self.clf = modules.ClassificationModule(d_model, factor, n_class)

    def forward(self, x, ts=None) -> torch.Tensor:
        x = self.encoder(x, ts)
        # x = self.dense_interpolation(x)
        x = self.clf(x)
        return x
