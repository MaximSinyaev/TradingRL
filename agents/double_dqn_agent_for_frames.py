import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

class QCNNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, num_frames: int, kernel_size: int=3):
        super().__init__()
        self.num_frames = num_frames
        self.num_features = state_dim
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=state_dim, out_channels=64, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(128 * (num_frames), 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        # X as input is (batch_size, num_frames, state_dim)
        x = x.permute(0, 2, 1) # (B, n_features, n_frames)
        x = self.conv(x)
        # print(f"X shape after conv: {x.shape}")
        return self.head(x)
    

class DoubleDQNCNNAgent:
    def __init__(
        self, state_dim, action_dim, num_frames, device=None, lr=7e-4, gamma=0.99,
        batch_size=64, buffer_size=100_000, epsilon_start=1.0,
        epsilon_end=0.1, epsilon_decay=0.995, gradient_clip=0.5,
        kernel_size=3, store_on_device: bool=False
    ):
        assert device in [None, "cpu", "cuda"], "device must be None, 'cpu' or 'cuda'"
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.store_on_device = store_on_device
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.gradient_clip = gradient_clip
        
        if 2 * (kernel_size - 1) >= num_frames:
            raise ValueError(f"kernel_size {kernel_size} is too large for num_frames {num_frames}.")

        self.q_network = QCNNNetwork(state_dim, action_dim, num_frames=num_frames, kernel_size=kernel_size)
        self.target_network = QCNNNetwork(state_dim, action_dim, num_frames=num_frames, kernel_size=kernel_size)
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        self.replay_buffer = deque(maxlen=buffer_size)
        self.td_errors = []

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.update_target_network()
        
    def reset_buffer(self):
        self.replay_buffer.clear()
        self.td_errors.clear()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_update(self, tau=0.005):
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_tensor = state.unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values).item())

    def store(self, state, action, reward, next_state, done):
        if self.store_on_device:
            state = state.to(self.device)
            next_state = next_state.to(self.device)
        
        self.replay_buffer.append((state, action, reward, next_state, done))

        # Рассчитываем TD-ошибку
        with torch.no_grad():
            state_tensor = state.unsqueeze(0).to(self.device)
            next_state_tensor = next_state.unsqueeze(0).to(self.device)
            action_tensor = torch.tensor([[action]])

            q_val = self.q_network(state_tensor).detach().cpu().gather(1, action_tensor)
            # Double DQN: выбираем действие по q_network, оцениваем по target_network
            next_action = self.q_network(next_state_tensor).argmax(1, keepdim=True)
            next_q_val = self.target_network(next_state_tensor).gather(1, next_action).detach().cpu()
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

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.q_network(states).gather(1, actions)
        # Double DQN: выбираем действие по q_network, оцениваем по target_network
        next_actions = self.q_network(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_network(next_states).gather(1, next_actions).detach()
        target = rewards + (1 - dones) * self.gamma * next_q_values

        # loss = F.smooth_l1_loss(q_values, target)
        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        self.optimizer.step()

        with torch.no_grad():
            q_values = self.q_network(states).gather(1, actions)
            # Double DQN: выбираем действие по q_network, оцениваем по target_network
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            td_errors_new = torch.abs(rewards + (1 - dones) * self.gamma * next_q_values - q_values).squeeze().tolist()
        for i, idx in enumerate(indices):
            self.td_errors[idx] = td_errors_new[i]

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.soft_update()