import torch
import torch.nn as nn

class RandomAgent:
    def __init__(self, action_dim: int = 3):
        self.q_network = self.RandomQNetwork(action_dim)

    class RandomQNetwork(nn.Module):
        def __init__(self, action_dim: int):
            super().__init__()
            self.action_dim = action_dim

        def forward(self, x):
            batch_size = x.size(0)
            # Генерация случайных Q-значений (логитов) для каждой пары (батч, действие)
            return torch.rand(batch_size, self.action_dim)

# Пример использования:
# random_agent = RandomAgent()
# validate_agent(env, random_agent, episodes=10)