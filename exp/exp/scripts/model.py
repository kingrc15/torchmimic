import torch
import torch.nn as nn

class LSTM_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 128
        
        self.lstm_layer = nn.LSTM(
                76,
                self.hidden_dim,
                num_layers=1,
                dropout=0.2,
                bidirectional=True,
        )
        
        self.final_layer = nn.Sequential(
            nn.Linear(128 * 2, 25),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 2, 1244, 128
        # 256, 1244, 76
        h0 = torch.zeros((2, x.size(1), self.hidden_dim)).float()
        c0 = torch.zeros((2, x.size(1), self.hidden_dim)).float()

        z, (hn, cn) = self.lstm_layer(x, (h0, c0))
        z = self.final_layer(z)
        
        return z