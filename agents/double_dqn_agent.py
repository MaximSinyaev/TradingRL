import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces
from agents.prioritized_replay_buffer import PrioritizedReplayBuffer


def _get_device(device=None):
    """Auto-detect best device if not specified."""
    if device is not None:
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon (M1/M2/M3)
    else:
        return torch.device("cpu")


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


class DoubleDQNAgent:
    def __init__(
        self, state_dim, action_dim=None, multi_discrete_actions=None,
        lr=1e-3, gamma=0.99, batch_size=64, buffer_size=100_000,
        epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
        grad_clip_norm=10.0, device=None,
        # Prioritized Replay Buffer params
        alpha=0.6, beta_start=0.4, beta_frames=100_000,
        use_prioritized_buffer=True,
    ):
        # Determine action dimensions
        if multi_discrete_actions is not None:
            self.multi_discrete = True
            self.action_dims = multi_discrete_actions
            action_dim = int(np.prod(multi_discrete_actions))
        else:
            self.multi_discrete = False
            self.action_dims = None
            if action_dim is None:
                raise ValueError("action_dim must be specified for Discrete action space")

        # Auto-detect device if not specified
        self.device = _get_device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip_norm = grad_clip_norm

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.use_prioritized_buffer = use_prioritized_buffer
        if use_prioritized_buffer:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_size,
                alpha=alpha,
                beta_start=beta_start,
                beta_frames=beta_frames,
            )
        else:
            self.replay_buffer = deque(maxlen=buffer_size)

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_update(self, tau=0.005):
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def _encode_action(self, action):
        """Encode action to int for internal use."""
        if self.multi_discrete:
            if isinstance(action, (list, np.ndarray)):
                return int(action[0] * self.action_dims[1] + action[1])
            return int(action)
        return int(action)

    def _decode_action(self, action_int):
        """Decode action from int to environment format."""
        if self.multi_discrete:
            return [action_int // self.action_dims[1], action_int % self.action_dims[1]]
        return int(action_int)

    def set_training_mode(self, training=True):
        """Set training or evaluation mode."""
        if training:
            self.q_network.train()
            self.target_network.train()
        else:
            self.q_network.eval()
            self.target_network.eval()

    def act(self, state, possible_actions=None, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            if possible_actions is not None and len(possible_actions) > 0:
                action = random.choice(possible_actions)
                action = self._encode_action(action) if isinstance(action, (list, np.ndarray)) else action
            else:
                action = random.randint(0, self.action_dim - 1)
            return self._decode_action(action)

        with torch.no_grad():
            # Convert numpy to torch if needed
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()
            state_tensor = state.unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)

            if possible_actions is not None and len(possible_actions) > 0:
                # Mask invalid actions (ensure mask is on same device as q_values)
                mask = torch.ones(self.action_dim, device=q_values.device) * float('-inf')
                for a in possible_actions:
                    encoded = self._encode_action(a)
                    mask[encoded] = 0
                q_values = q_values + mask.unsqueeze(0)

            action_int = int(torch.argmax(q_values).item())
            return self._decode_action(action_int)

    def decay_epsilon(self):
        """Decay epsilon (call at end of episode, not every step!)."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def store(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        # Encode action for MultiDiscrete
        action_encoded = self._encode_action(action)

        if self.use_prioritized_buffer:
            # PrioritizedReplayBuffer handles conversion internally
            self.replay_buffer.push(state, action_encoded, reward, next_state, done)
        else:
            # Convert numpy arrays to torch tensors for storage
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()
            if isinstance(next_state, np.ndarray):
                next_state = torch.from_numpy(next_state).float()

            self.replay_buffer.append((state, action_encoded, reward, next_state, done))

    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        if self.use_prioritized_buffer:
            # Sample from PrioritizedReplayBuffer
            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)

            # Move to device
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            weights = weights.to(self.device)

            # Compute current Q values
            q_values = self.q_network(states).gather(1, actions)

            # Double DQN: use q_network to select actions, target_network to evaluate
            with torch.no_grad():
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
                target = rewards + (1 - dones) * self.gamma * next_q_values

            # Compute TD errors for priority update
            td_errors = torch.abs(target - q_values)

            # Compute weighted loss (importance sampling)
            loss = (weights * F.mse_loss(q_values, target, reduction='none')).mean()

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip_norm)

            self.optimizer.step()

            # Update priorities in buffer
            self.replay_buffer.update_priorities(indices, td_errors)

        else:
            # Fallback: uniform sampling
            batch = random.sample(list(self.replay_buffer), self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.stack(states).to(self.device)
            actions = torch.tensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
            next_states = torch.stack(next_states).to(self.device)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

            # Compute current Q values
            q_values = self.q_network(states).gather(1, actions)

            # Double DQN: use q_network to select actions, target_network to evaluate
            with torch.no_grad():
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
                target = rewards + (1 - dones) * self.gamma * next_q_values

            # Compute loss
            loss = F.mse_loss(q_values, target)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip_norm)

            self.optimizer.step()

        # Soft update target network
        self.soft_update()

        return loss.item()

    def save_state_dict(self, path):
        """Save model state."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load_state_dict(self, path):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

    @classmethod
    def from_env(cls, env, **kwargs):
        """Create agent from environment (auto-detect action space).

        Usage:
            agent = DoubleDQNAgent.from_env(env, lr=5e-4)  # Custom lr
        """
        # Detect action space type
        if isinstance(env.action_space, spaces.MultiDiscrete):
            multi_discrete = env.action_space.nvec.tolist()
            action_dim = int(np.prod(multi_discrete))
            state_dim = env.observation_space.shape[0]

            # Override if not specified
            if 'multi_discrete_actions' not in kwargs:
                kwargs['multi_discrete_actions'] = multi_discrete
            if 'action_dim' in kwargs:
                del kwargs['action_dim']  # Not needed for MultiDiscrete

            return cls(state_dim=state_dim, action_dim=action_dim, **kwargs)

        elif isinstance(env.action_space, spaces.Discrete):
            action_dim = env.action_space.n
            state_dim = env.observation_space.shape[0]

            return cls(state_dim=state_dim, action_dim=action_dim, **kwargs)

        else:
            raise ValueError(f"Unsupported action space: {type(env.action_space)}")
