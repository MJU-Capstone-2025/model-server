import torch
import numpy as np

class MultiStepTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, window_size, step, static_idx):
        self.X_seq = []
        self.X_static = []
        self.y = []
        self.seq_idx = [i for i in range(X.shape[1]) if i not in static_idx]

        for i in range(0, len(X) - window_size, step):
            if np.isnan(y[i + window_size]): continue
            self.X_seq.append(X[i:i+window_size, self.seq_idx])
            self.X_static.append(X[i, static_idx])
            self.y.append(y[i + window_size])

        self.X_seq = torch.tensor(self.X_seq, dtype=torch.float32)
        self.X_static = torch.tensor(self.X_static, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X_seq[idx], self.X_static[idx], self.y[idx]