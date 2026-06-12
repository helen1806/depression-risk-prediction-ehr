import torch
import torch.nn as nn

class RFApproximationMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3):
        super(RFApproximationMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))
