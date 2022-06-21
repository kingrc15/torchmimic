import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM_Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        if args.bidirectional:
            self.hidden_dim = args.hidden_dim // 2

        self.lstm_layer = nn.LSTM(
            76,
            self.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout_rate,
            bidirectional=args.bidirectional,
            batch_first=True,
        )

        linear_input = self.hidden_dim
        if args.bidirectional:
            linear_input *= 2

        self.final_layer = nn.Sequential(
            nn.Dropout(args.dropout_rate),
            nn.Linear(linear_input, args.n_classes),
            nn.Sigmoid()
        )

    def forward(self, data):
        seq = data[0]
        lens = data[1]
        packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)

        z, (hn, cn) = self.lstm_layer(packed)

        seq_unpacked, lens_unpacked = pad_packed_sequence(z, batch_first=True)
        output = self.final_layer(
            torch.vstack([seq_unpacked[i, int(l) - 1] for i, l in enumerate(lens)])
        )

        return output
