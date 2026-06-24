"""
Prioritized Experience Replay Buffer.

Based on "Prioritized Experience Replay" (Schaul et al., 2016).
Uses proportional prioritization with importance sampling.

Key features:
- Proportional prioritization (priority_i ~ |TD_error|^alpha)
- Importance sampling weights (IS) to correct bias
- Annealing beta from 0.4 to 1.0
- Efficient O(log n) operations with SumTree
"""

import numpy as np
import torch


class SumTree:
    """Binary tree for O(log n) priority updates and sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0
        self.filled_slots = set()  # Track which slots have data

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample index given cumulative priority value."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Return total priority."""
        return self.tree[0]

    def add(self, priority: float, data):
        """Add new experience with priority."""
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.filled_slots.add(self.write)  # Mark as filled
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float):
        """Update priority at index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> tuple:
        """Get (index, priority, data) given cumulative priority."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        # Retry if hit empty slot (shouldn't happen with filled_slots tracking)
        if data_idx not in self.filled_slots:
            # Fallback: try to get nearest filled slot
            for offset in range(1, min(10, self.capacity)):
                for direction in [-1, 1]:
                    test_idx = (data_idx + direction * offset) % self.capacity
                    if test_idx in self.filled_slots:
                        # Recompute idx for this data_idx
                        tree_idx = test_idx + self.capacity - 1
                        return (tree_idx, self.tree[tree_idx], self.data[test_idx])
            raise ValueError(f"Could not find valid data slot near {data_idx}")

        data = self.data[data_idx]
        return (idx, self.tree[idx], data)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer with importance sampling.

    Args:
        capacity: Max number of experiences
        alpha: Priority exponent (0=uniform, 1=full prioritization)
        beta_start: Initial IS weight (0=no correction, 1=full correction)
        beta_frames: Anneal beta to 1.0 over this many frames
        epsilon: Small constant to ensure non-zero priorities
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        epsilon: float = 1e-6
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 0

        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        """Store experience with max priority (new experiences are important)."""
        # Convert to tensors if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state).float()

        experience = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority, experience)

    def sample(self, batch_size: int) -> tuple:
        """Sample batch proportional to priorities.

        Returns:
            (states, actions, rewards, next_states, dones, indices, weights)
            - weights: Importance sampling weights for loss correction
        """
        if len(self) < batch_size:
            raise ValueError(f"Not enough samples in buffer: {len(self)}/{batch_size}")

        batch_data = []
        indices = []
        priorities = []

        # Split total priority into batch_size segments
        segment = self.tree.total() / batch_size

        # Update beta (anneal to 1.0)
        self.frame += 1
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) *
                   (self.frame / self.beta_frames))

        for i in range(batch_size):
            # Sample from each segment
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)

            idx, priority, data = self.tree.get(s)

            # Skip None entries (shouldn't happen with proper n_entries)
            if data is None:
                continue

            batch_data.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Ensure we have enough samples
        if len(batch_data) < batch_size:
            raise ValueError(f"Not enough valid samples: {len(batch_data)}/{batch_size}")

        # Unpack
        states, actions, rewards, next_states, dones = zip(*batch_data)

        # Compute importance sampling weights
        priorities = np.array(priorities)
        probabilities = priorities / self.tree.total()
        weights = (self.capacity * probabilities) ** (-beta)
        weights = weights / weights.max()  # Normalize

        # Convert to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

        return states, actions, rewards, next_states, dones, indices, weights_tensor

    def update_priorities(self, indices: list, td_errors: torch.Tensor):
        """Update priorities based on TD errors.

        Args:
            indices: Tree indices from sample()
            td_errors: TD errors from training step
        """
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()

        # Priority = |TD_error|^alpha + epsilon
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha

        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)

        # Track max priority for new experiences
        self.max_priority = max(self.max_priority, priorities.max())

    def __len__(self) -> int:
        return self.tree.n_entries
