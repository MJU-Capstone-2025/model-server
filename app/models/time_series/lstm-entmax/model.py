import torch
import torch.nn as nn
from entmax import Entmax15
import numpy as np
import pandas as pd

class EntmaxAttention(nn.Module):
    def __init__(self, hidden_size, attn_dim=128):
        super().__init__()
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_size, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1))
        self.entmax = Entmax15(dim=1)

    def forward(self, x):
        scores = self.score_layer(x).squeeze(-1)
        weights = self.entmax(scores)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context, weights

class AttentionLSTMModel(nn.Module):
    def __init__(self, input_size, static_dim, hidden=128, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.attn = EntmaxAttention(hidden)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, 32), nn.ReLU(), nn.LayerNorm(32),
            nn.Dropout(0.2), nn.Linear(32, 64), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(hidden*2, 1), nn.Sigmoid())
        self.fc = nn.Sequential(
            nn.Linear(hidden + 64, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x_seq, x_static):
        lstm_out, _ = self.lstm(x_seq)
        context, _ = self.attn(lstm_out)
        last = lstm_out[:, -1, :]
        alpha = self.gate(torch.cat([context, last], dim=1))
        fused = alpha * context + (1 - alpha) * last
        static_enc = self.static_encoder(x_static)
        return self.fc(torch.cat([fused, static_enc], dim=1)).squeeze(-1)
