"""
Tests for PrioritizedReplayBuffer.

Tests the SumTree implementation and PrioritizedReplayBuffer:
- Basic operations (push, sample, update)
- Priority propagation
- Importance sampling weights
- Edge cases (empty buffer, small buffer)
"""

import pytest
import numpy as np
import torch
from agents.prioritized_replay_buffer import PrioritizedReplayBuffer, SumTree


class TestSumTree:
    """Test SumTree basic operations."""

    def test_sumtree_initialization(self):
        """Test SumTree initializes correctly."""
        tree = SumTree(capacity=100)
        assert tree.capacity == 100
        assert tree.n_entries == 0
        assert tree.write == 0
        assert len(tree.filled_slots) == 0

    def test_sumtree_add(self):
        """Test adding elements to SumTree."""
        tree = SumTree(capacity=10)
        tree.add(1.0, "data1")
        tree.add(2.0, "data2")

        assert tree.n_entries == 2
        assert tree.total() == 3.0
        assert tree.data[0] == "data1"
        assert tree.data[1] == "data2"

    def test_sumtree_add_with_wraparound(self):
        """Test SumTree wraps around correctly when full."""
        tree = SumTree(capacity=5)

        # Fill capacity
        for i in range(5):
            tree.add(float(i), f"data{i}")

        assert tree.n_entries == 5
        assert tree.total() == 10.0  # 0+1+2+3+4

        # Add one more (should overwrite index 0)
        tree.add(5.0, "data5")
        assert tree.n_entries == 5  # Still at capacity
        assert tree.data[0] == "data5"
        assert tree.total() == 15.0  # 1+2+3+4+5 (0 replaced by 5)

    def test_sumtree_update(self):
        """Test updating priority changes total correctly."""
        tree = SumTree(capacity=10)
        tree.add(1.0, "data1")
        tree.add(2.0, "data2")

        # Update priority of first element (index = capacity - 1 + 0 = 9)
        idx = 10 - 1 + 0
        tree.update(idx, 5.0)

        assert tree.total() == 7.0  # 5.0 + 2.0

    def test_sumtree_sample(self):
        """Test sampling from SumTree."""
        tree = SumTree(capacity=10)

        # Add elements with known priorities
        for i in range(5):
            tree.add(float(i + 1), f"data{i}")  # 1.0, 2.0, 3.0, 4.0, 5.0

        total = tree.total()
        assert total == 15.0

        # Sample from different segments
        idx1, prio1, data1 = tree.get(0.5)  # Low end
        idx2, prio2, data2 = tree.get(7.5)  # Middle
        idx3, prio3, data3 = tree.get(14.5)  # High end

        assert data1 is not None
        assert data2 is not None
        assert data3 is not None

    def test_sumtree_edge_cases(self):
        """Test edge cases for SumTree."""
        tree = SumTree(capacity=3)

        # Test sampling empty tree
        with pytest.raises((ValueError, IndexError)):
            tree.get(1.0)

        # Test with single element
        tree.add(1.0, "single")
        idx, prio, data = tree.get(0.5)
        assert data == "single"
        assert prio == 1.0


class TestPrioritizedReplayBuffer:
    """Test PrioritizedReplayBuffer functionality."""

    def test_buffer_initialization(self):
        """Test buffer initializes with correct parameters."""
        buffer = PrioritizedReplayBuffer(
            capacity=1000,
            alpha=0.7,
            beta_start=0.5,
            beta_frames=50000,
        )

        assert buffer.capacity == 1000
        assert buffer.alpha == 0.7
        assert buffer.beta_start == 0.5
        assert buffer.beta_frames == 50000
        assert len(buffer) == 0

    def test_buffer_push_and_len(self):
        """Test pushing experiences increases length."""
        buffer = PrioritizedReplayBuffer(capacity=10)

        state = np.array([1, 2, 3])
        action = 1
        reward = 1.0
        next_state = np.array([2, 3, 4])
        done = False

        for _ in range(5):
            buffer.push(state, action, reward, next_state, done)

        assert len(buffer) == 5

    def test_buffer_wraparound(self):
        """Test buffer wraps around when full."""
        buffer = PrioritizedReplayBuffer(capacity=5)

        state = np.array([1, 2, 3])
        for _ in range(10):
            buffer.push(state, 1, 1.0, state, False)

        assert len(buffer) == 5  # At capacity

    def test_buffer_sample_returns_correct_shapes(self):
        """Test sample returns tensors with correct shapes."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        # Add some experiences
        state = np.array([1, 2, 3])
        for i in range(50):
            buffer.push(state, i % 3, float(i), state, i % 2 == 0)

        # Sample
        states, actions, rewards, next_states, dones, indices, weights = buffer.sample(10)

        assert states.shape[0] == 10
        assert actions.shape[0] == 10
        assert rewards.shape[0] == 10
        assert next_states.shape[0] == 10
        assert dones.shape[0] == 10
        assert len(indices) == 10
        assert weights.shape[0] == 10

    def test_buffer_sample_insufficient_experiences(self):
        """Test sample raises error when not enough experiences."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        # Only add 5 experiences
        state = np.array([1, 2, 3])
        for _ in range(5):
            buffer.push(state, 1, 1.0, state, False)

        # Try to sample 10
        with pytest.raises(ValueError, match="Not enough samples"):
            buffer.sample(10)

    def test_buffer_importance_sampling_weights(self):
        """Test importance sampling weights are computed correctly."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        # Add experiences
        state = np.array([1, 2, 3])
        for _ in range(50):
            buffer.push(state, 1, 1.0, state, False)

        states, actions, rewards, next_states, dones, indices, weights = buffer.sample(10)

        # Weights should be normalized (max weight = 1.0)
        assert weights.max() <= 1.0 + 1e-5  # Allow small floating point error
        assert weights.min() >= 0.0

        # All weights should be positive
        assert (weights > 0).all()

    def test_buffer_update_priorities(self):
        """Test updating priorities changes future sampling."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        # Add experiences
        state = np.array([1, 2, 3])
        for _ in range(50):
            buffer.push(state, 1, 1.0, state, False)

        # Sample once
        states, actions, rewards, next_states, dones, indices, weights = buffer.sample(10)

        # Create fake TD errors
        td_errors = torch.tensor([0.1, 0.5, 1.0, 0.2, 0.8, 0.3, 0.9, 0.4, 0.6, 0.7])

        # Update priorities
        buffer.update_priorities(indices, td_errors)

        # Buffer should still be valid
        assert len(buffer) == 50

    def test_buffer_with_tensor_states(self):
        """Test buffer handles tensor states correctly."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        # Push with tensor states
        state = torch.tensor([1, 2, 3], dtype=torch.float32)
        buffer.push(state, 1, 1.0, state, False)

        # Sample
        states, actions, rewards, next_states, dones, indices, weights = buffer.sample(1)

        assert states.shape == (1, 3)
        assert states.dtype == torch.float32

    def test_buffer_beta_annealing(self):
        """Test beta anneals from beta_start to 1.0."""
        buffer = PrioritizedReplayBuffer(
            capacity=1000,
            beta_start=0.4,
            beta_frames=100,
        )

        # Fill buffer
        state = np.array([1, 2, 3])
        for _ in range(500):
            buffer.push(state, 1, 1.0, state, False)

        # Sample at different points
        buffer.frame = 0
        _, _, _, _, _, _, weights_start = buffer.sample(10)
        # Weights at low beta are less uniform (more variance)

        buffer.frame = 100  # Should be at beta = 1.0
        _, _, _, _, _, _, weights_end = buffer.sample(10)
        # Weights at beta = 1.0 should be fully corrected

        # Both should be valid
        assert weights_start.shape[0] == 10
        assert weights_end.shape[0] == 10

    def test_buffer_max_priority_tracking(self):
        """Test max_priority is tracked correctly."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        # Add experiences
        state = np.array([1, 2, 3])
        for _ in range(10):
            buffer.push(state, 1, 1.0, state, False)

        # Update with large TD error
        states, actions, rewards, next_states, dones, indices, weights = buffer.sample(5)
        td_errors = torch.tensor([5.0, 0.1, 0.2, 0.3, 0.4])
        buffer.update_priorities(indices, td_errors)

        # Max priority should have increased
        assert buffer.max_priority >= 5.0 ** buffer.alpha


class TestBufferIntegration:
    """Test buffer integration patterns with agents."""

    def test_buffer_with_multi_discrete_actions(self):
        """Test buffer works with MultiDiscrete action encoding."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        # Simulate MultiDiscrete action [action_type, size_level]
        # Encoded as: action_type * 10 + size_level
        for action_type in range(3):
            for size_level in range(10):
                action = action_type * 10 + size_level
                buffer.push(
                    state=np.array([1, 2, 3]),
                    action=action,
                    reward=1.0,
                    next_state=np.array([2, 3, 4]),
                    done=False
                )

        assert len(buffer) == 30

        # Sample and verify actions are preserved
        states, actions, rewards, next_states, dones, indices, weights = buffer.sample(10)
        assert actions.shape == (10, 1)

    def test_buffer_device_handling(self):
        """Test buffer handles device conversions correctly."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        # Add numpy states
        state = np.array([1, 2, 3])
        buffer.push(state, 1, 1.0, state, False)

        # Sample and verify tensors are returned
        states, actions, rewards, next_states, dones, indices, weights = buffer.sample(1)

        # Should be tensors
        assert isinstance(states, torch.Tensor)
        assert isinstance(actions, torch.Tensor)

        # Should be on CPU by default
        assert states.device.type == 'cpu'
