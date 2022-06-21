import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import modules


class EncoderLayerForSAnD(nn.Module):
    def __init__(
        self, input_features, seq_len, n_heads, n_layers, d_model=128, dropout_rate=0.2
    ) -> None:
        super(EncoderLayerForSAnD, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.input_embedding = nn.Conv1d(input_features, d_model, 1)
        self.positional_encoding = modules.PositionalEncoding(d_model, seq_len)
        print(n_layers)
        self.blocks = nn.ModuleList([
            modules.EncoderBlock(d_model, n_heads, dropout_rate) for _ in range(n_layers)
        ])
    def forward(self, x, mask) -> torch.Tensor:
        x[mask] = 0
        x = x.transpose(1, 2)
        x = self.input_embedding(x)
        x = x.transpose(1, 2)
        
        x = self.positional_encoding(x)
        
        mask = torch.cat((mask, torch.ones((x.size(0), self.seq_len - x.size(1)), device = x.device)), dim=1).bool()
        x = torch.cat((x, torch.zeros(x.size(0), self.seq_len - x.size(1), x.size(2), device = x.device)), dim =1)

        for l in self.blocks:
            x = l(x, mask)

        return x


class SAnD(nn.Module):
    """
    Simply Attend and Diagnose model

    The Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18)

    `Attend and Diagnose: Clinical Time Series Analysis Using Attention Models <https://arxiv.org/abs/1711.03905>`_
    Huan Song, Deepta Rajan, Jayaraman J. Thiagarajan, Andreas Spanias
    """

    def __init__(
        self,
        input_features: int,
        seq_len: int,
        n_heads: int,
        factor: int,
        n_class: int,
        n_layers: int,
        d_model: int = 128,
        dropout_rate: float = 0.2,
    ) -> None:
        super(SAnD, self).__init__()
        self.seq_len = 500
        self.encoder = EncoderLayerForSAnD(
            input_features, seq_len, n_heads, n_layers, d_model, dropout_rate
        )
        self.dense_interpolation = modules.DenseInterpolation(seq_len, factor)
        self.clf = modules.ClassificationModule(d_model, factor, n_class)

    def forward(self, data) -> torch.Tensor:
        x = data[0][:,:self.seq_len]
        mask = data[3][:,:self.seq_len].to(x.device)
        
        x = self.encoder(x, mask)
        x = self.dense_interpolation(x)
        x = self.clf(x)
        return x
