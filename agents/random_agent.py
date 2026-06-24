import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class RandomAgent:
    """Random agent for baseline comparisons."""

    def __init__(self, action_dim: int = 3, seed: Optional[int] = None):
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed) if seed else np.random

    def act(self, state, possible_actions=None):
        """Select random action.

        Args:
            state: Current state (ignored for random agent)
            possible_actions: List of valid actions (optional)
                             For MultiDiscrete: list of [action, size] pairs

        Returns:
            Random action (int or np.ndarray)
        """
        if possible_actions is not None:
            choice = self.rng.choice(possible_actions)
            # Handle both Discrete (int) and MultiDiscrete (list)
            if isinstance(choice, (list, np.ndarray)):
                return np.array(choice)
            return int(choice)

        action = self.rng.integers(0, self.action_dim)
        # Return as scalar for Discrete, as array for MultiDiscrete
        return action if self.action_dim <= 3 else np.array([action, 0])

    def update(self, state, action, reward, next_state, done):
        """No-op for random agent (no learning)."""
        pass
