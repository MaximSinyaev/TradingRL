import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input):
        if self.training and input.size(0) == 1:
            # Пропускаем нормализацию, возвращаем вход как есть
            return input
        return super().forward(input)

class QNetwork(nn.Module):
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

class DQNAgent:
    def __init__(
        self, state_dim, action_dim, lr=1e-3, gamma=0.99,
        batch_size=64, buffer_size=100_000, epsilon_start=1.0,
        epsilon_end=0.1, epsilon_decay=0.995
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.q_network.to(device)
        self.target_network.to(device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        self.replay_buffer = deque(maxlen=buffer_size)
        self.td_errors = []

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_update(self, tau=0.005):
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values).item())

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

        # Рассчитываем TD-ошибку
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            action_tensor = torch.tensor([[action]])

            q_val = self.q_network(state_tensor).detach().cpu().gather(1, action_tensor)
            next_q_val = self.target_network(next_state_tensor).max(1, keepdim=True)[0].detach().cpu()
            td_error = torch.abs(reward + (1 - done) * self.gamma * next_q_val - q_val).item()
        
        self.td_errors.append(td_error)

        if len(self.td_errors) > self.replay_buffer.maxlen:
            self.td_errors.pop(0)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        td_error_np = np.array(self.td_errors)
        probs = td_error_np / td_error_np.sum()
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, p=probs)
        batch = [self.replay_buffer[i] for i in indices]

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1, keepdim=True)[0].detach()
        target = rewards + (1 - dones) * self.gamma * next_q_values

        # loss = F.smooth_l1_loss(q_values, target)
        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            q_values = self.q_network(states).gather(1, actions)
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            td_errors_new = torch.abs(rewards + (1 - dones) * self.gamma * next_q_values - q_values).squeeze().tolist()
        for i, idx in enumerate(indices):
            self.td_errors[idx] = td_errors_new[i]

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.soft_update()