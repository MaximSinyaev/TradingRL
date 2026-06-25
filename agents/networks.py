import torch
import torch.nn as nn

class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input):
        if self.training and input.size(0) == 1:
            return input
        return super().forward(input)


class QNetwork(nn.Module):
    """Q-Network for DQN."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)
