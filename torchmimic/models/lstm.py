import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class StandardLSTM(nn.Module):
    def __init__(
        self,
        n_classes,
        hidden_dim=128,
        num_layers=1,
        dropout_rate=0,
        bidirectional=False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional

        if bidirectional:
            self.hidden_dim = hidden_dim // 2

        if num_layers == 1:
            dropout_rate = 0

        self.lstm_layer = nn.LSTM(
            76,
            self.hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate,
            bidirectional=bidirectional,
            batch_first=True,
        )

        linear_input = self.hidden_dim
        if bidirectional:
            linear_input *= 2

        self.final_layer = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(linear_input, n_classes),
            nn.Sigmoid(),
        )

    def forward(self, data):
        seq = data[0]
        lens = data[1]
        packed = pack_padded_sequence(
            seq, lens, batch_first=True, enforce_sorted=False
        )

        h_dim = 2 if self.bidirectional else 1

        z, (ht, ct) = self.lstm_layer(packed)

        seq_unpacked, lens_unpacked = pad_packed_sequence(z, batch_first=True)

        output = self.final_layer(
            torch.vstack(
                [
                    seq_unpacked[i, int(l) - 1]
                    for i, l in enumerate(lens_unpacked)
                ]
            )
        )

        return output

    def get_config(self):
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "bidirectional": self.bidirectional,
        }
