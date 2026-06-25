"""
Universal DQN Agent for Trading Environments.

Supports:
- Discrete action space: V1, V2, V3
- MultiDiscrete action space: V4 (and future)

Usage:
    # For Discrete (V1, V2, V3)
    agent = DQNAgent(state_dim=23, action_dim=3)

    # For MultiDiscrete (V4)
    agent = DQNAgent(state_dim=25, action_dim=None, multi_discrete_actions=[3, 10])

    # Auto-detect from env
    agent = DQNAgent.from_env(env)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from typing import Optional, Union, List, Tuple
from gymnasium import spaces
from agents.prioritized_replay_buffer import PrioritizedReplayBuffer
from agents.networks import QNetwork


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    """Universal DQN Agent with Discrete and MultiDiscrete support.

    Default hyperparameters optimized for trading environments:
    - lr=1e-3
    - gamma=0.99
    - batch_size=64
    - epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995
    - buffer_size=100_000

    For MultiDiscrete actions (e.g., V4):
        - Flattens to Discrete internally
        - Returns [action_type, size_level] from act()
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: Optional[int] = None,
        multi_discrete_actions: Optional[List[int]] = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        # Prioritized Replay Buffer params
        alpha: float = 0.6,  # Priority exponent
        beta_start: float = 0.4,  # IS weight start
        beta_frames: int = 100_000,  # Anneal IS to 1.0
        # Customizable for faster/slower learning
        tau: float = 0.005,  # Soft update rate
        use_prioritized_buffer: bool = True,
    ):
        # Determine action dimensions
        if multi_discrete_actions is not None:
            # MultiDiscrete: [3, 10] → 30 total actions
            self.multi_discrete = True
            self.action_dims = multi_discrete_actions
            action_dim = int(np.prod(multi_discrete_actions))
        else:
            # Discrete
            self.multi_discrete = False
            self.action_dims = None
            if action_dim is None:
                raise ValueError("action_dim must be specified for Discrete action space")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        # Networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.q_network.to(device)
        self.target_network.to(device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer with prioritized experience sampling
        self.use_prioritized_buffer = use_prioritized_buffer
        if use_prioritized_buffer:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_size,
                alpha=alpha,
                beta_start=beta_start,
                beta_frames=beta_frames,
            )
            self.td_errors = None  # Not needed with PER
        else:
            # Fallback to simple deque (for compatibility)
            self.replay_buffer = deque(maxlen=buffer_size)
            self.td_errors = deque(maxlen=buffer_size)

        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.update_target_network()

    @classmethod
    def from_env(cls, env, **kwargs):
        """Create agent from environment (auto-detect action space).

        Usage:
            agent = DQNAgent.from_env(env, lr=5e-4)  # Custom lr
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

    def _encode_action(self, action: Union[int, List[int]]) -> int:
        """Encode action to int for internal use.

        Args:
            action: int (Discrete) or [action_type, size_level] (MultiDiscrete)

        Returns:
            int: Flattened action index
        """
        if self.multi_discrete:
            # [action_type, size_level] → int
            if isinstance(action, (list, np.ndarray)):
                return int(action[0] * self.action_dims[1] + action[1])
            return int(action)  # Already encoded
        return int(action)

    def _decode_action(self, action_int: int) -> Union[int, List[int]]:
        """Decode action from int to environment format.

        Args:
            action_int: Flattened action index

        Returns:
            int (Discrete) or [action_type, size_level] (MultiDiscrete)
        """
        if self.multi_discrete:
            # int → [action_type, size_level]
            return [action_int // self.action_dims[1], action_int % self.action_dims[1]]
        return int(action_int)

    def act(self, state, possible_actions=None, training=True) -> Union[int, List[int]]:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state
            possible_actions: Optional mask for valid actions
                - For Discrete: list of valid ints
                - For MultiDiscrete: list of [action_type, size_level]
            training: If False, disable exploration

        Returns:
            int or [action_type, size_level]
        """
        if training and random.random() < self.epsilon:
            # Explore
            if possible_actions is not None and len(possible_actions) > 0:
                action = random.choice(possible_actions)
            else:
                action = random.randint(0, self.action_dim - 1)
            return self._decode_action(action)

        # Exploit
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)

            # Mask invalid actions
            if possible_actions is not None:
                mask = torch.full_like(q_values, float('-inf'))
                for action in possible_actions:
                    idx = self._encode_action(action)
                    mask[0, idx] = q_values[0, idx]
                q_values = mask

            action_int = int(torch.argmax(q_values).item())
            return self._decode_action(action_int)

    def store(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        # Encode action for storage
        action_encoded = self._encode_action(action)

        if self.use_prioritized_buffer:
            # PrioritizedReplayBuffer handles priorities internally
            self.replay_buffer.push(state, action_encoded, reward, next_state, done)
        else:
            # Fallback: simple deque
            self.replay_buffer.append((state, action_encoded, reward, next_state, done))

            # Calculate TD-error for prioritized sampling
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
                action_tensor = torch.tensor([[action_encoded]])

                q_val = self.q_network(state_tensor).detach().cpu().gather(1, action_tensor)
                next_q_val = self.target_network(next_state_tensor).max(1, keepdim=True)[0].detach().cpu()
                td_error = torch.abs(reward + (1 - done) * self.gamma * next_q_val - q_val).item()

            self.td_errors.append(td_error)

    def train_step(self) -> Optional[float]:
        """Train Q-network on one batch.

        Returns:
            Loss value or None if buffer too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        if self.use_prioritized_buffer:
            # Sample from PrioritizedReplayBuffer
            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)

            # Move to device
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)
            weights = weights.to(device)

            # Compute Q-values
            q_values = self.q_network(states).gather(1, actions)
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0].detach()
            target = rewards + (1 - dones) * self.gamma * next_q_values

            # Compute weighted loss (importance sampling)
            td_errors = torch.abs(target - q_values)
            loss = (weights * F.mse_loss(q_values, target, reduction='none')).mean()

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()

            # Update priorities in buffer based on new TD-errors
            self.replay_buffer.update_priorities(indices, td_errors)

        else:
            # Fallback: naive O(N) sampling (for compatibility)
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

            # Compute Q-values
            q_values = self.q_network(states).gather(1, actions)
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0].detach()
            target = rewards + (1 - dones) * self.gamma * next_q_values

            # Compute loss
            loss = F.mse_loss(q_values, target)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()

            # Update TD-errors
            with torch.no_grad():
                q_values_new = self.q_network(states).gather(1, actions)
                next_q_values_new = self.target_network(next_states).max(1, keepdim=True)[0]
                td_errors_new = torch.abs(rewards + (1 - dones) * self.gamma * next_q_values_new - q_values_new).squeeze().tolist()
            for i, idx in enumerate(indices):
                self.td_errors[idx] = td_errors_new[i]

        # Update exploration
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Soft update target network
        self.soft_update()

        return loss.item()

    def update_target_network(self):
        """Hard update of target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_update(self):
        """Soft update of target network."""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path: str):
        """Save model."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)
        print(f"✅ Model saved to {path}")

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        print(f"✅ Model loaded from {path}")
