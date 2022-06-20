import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(seq_len, d_model)

        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x) -> torch.Tensor:
        seq_len = x.shape[1]
        x = math.sqrt(self.d_model) * x
        x = x + self.pe[:, :seq_len].requires_grad_(False)
        return x


class TimeEncoding(nn.Module):
    def __init__(self, d_model, seq_len) -> None:
        super(TimeEncoding, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(seq_len, d_model)

        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x, ts) -> torch.Tensor:
        pe = torch.zeros(seq_len, d_model)
        for pos in enumerate(ts):
            for i in range(0, self.d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        seq_len = x.shape[1]
        x = math.sqrt(self.d_model) * x
        x = x + pe[:, :seq_len].requires_grad_(False)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, layer: nn.Module, embed_dim: int, p=0.1) -> None:
        super(ResidualBlock, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(p=p)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [N, seq_len, features]
        :return: [N, seq_len, features]
        """
        if isinstance(self.layer, nn.MultiheadAttention):
            src = x.transpose(0, 1)  # [seq_len, N, features]
            output, self.attn_weights = self.layer(src, src, src)
            output = output.transpose(0, 1)  # [N, seq_len, features]

        else:
            output = self.layer(x)

        output = self.dropout(output)
        output = self.norm(x + output)
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.hidden_size = hidden_size

        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size * 2, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_size * 2, hidden_size, 1),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.transpose(1, 2)
        tensor = self.conv(tensor)
        tensor = tensor.transpose(1, 2)

        return tensor


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_head: int, dropout_rate=0.1) -> None:
        super(EncoderBlock, self).__init__()
        self.attention = ResidualBlock(
            nn.MultiheadAttention(embed_dim, num_head), embed_dim, p=dropout_rate
        )
        self.ffn = ResidualBlock(
            PositionWiseFeedForward(embed_dim), embed_dim, p=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.attention(x)
        x = self.ffn(x)
        return x + identity


class DenseInterpolation(nn.Module):
    def __init__(self, seq_len: int, factor: int) -> None:
        """
        :param seq_len: sequence length
        :param factor: factor M
        """
        super(DenseInterpolation, self).__init__()

        W = np.zeros((factor, seq_len), dtype=np.float32)

        for t in range(seq_len):
            s = np.array((factor * (t + 1)) / seq_len, dtype=np.float32)
            for m in range(factor):
                tmp = np.array(1 - (np.abs(s - (1 + m)) / factor), dtype=np.float32)
                w = np.power(tmp, 2, dtype=np.float32)
                W[m, t] = w

        W = torch.tensor(W).float().unsqueeze(0)
        self.register_buffer("W", W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.W.repeat(x.shape[0], 1, 1).requires_grad_(False)
        u = torch.bmm(w, x)
        return u.transpose_(1, 2)


class ClassificationModule(nn.Module):
    def __init__(self, d_model: int, factor: int, num_class: int) -> None:
        super(ClassificationModule, self).__init__()
        self.d_model = d_model
        self.factor = factor
        self.num_class = num_class

        self.fc = nn.Sequential(
            nn.Linear(int(d_model * factor), num_class), nn.Softmax(dim=1)
        )

        # nn.init.normal_(self.fc[0].weight, std=0.02)
        # nn.init.normal_(self.fc[0].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, int(self.factor * self.d_model))
        x = self.fc(x)
        return x


class Attn(nn.Module):
    def __init__(self, embedding_size, num_heads, qk_scale=None, qkv_bias=None):
        super().__init__()

        self.num_heads = num_heads
        head_dim = embedding_size // num_heads
        all_head_dim = head_dim * num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(embedding_size, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.proj = nn.Linear(all_head_dim, embedding_size)

    def forward(self, x):

        B, N, C = x.shape
        qkv_bias = None
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, -1).permute(
            2, 0, 3, 1, 4
        )  # [B, NH, -1]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)

        return x


class Block(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()
        self.attn = Attn(embedding_size, num_heads)
        self.mlp = nn.Linear(embedding_size, embedding_size)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class RegressionModule(nn.Module):
    def __init__(self, d_model: int, factor: int, output_size: int) -> None:
        super(RegressionModule, self).__init__()
        self.d_model = d_model
        self.factor = factor
        self.output_size = output_size
        self.fc = nn.Linear(int(d_model * factor), output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().view(-1, int(self.factor * self.d_model))
        x = self.fc(x)
        return x
