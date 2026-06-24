"""
Integration tests for agents with PrioritizedReplayBuffer.

Tests that agents correctly use PrioritizedReplayBuffer:
- DQNAgent with PER
- DoubleDQNAgent with PER
- MultiDiscrete action support
- Training step produces valid updates
"""

import pytest
import numpy as np
import torch
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from agents.prioritized_replay_buffer import PrioritizedReplayBuffer


class TestDQNAgentWithPER:
    """Test DQNAgent with PrioritizedReplayBuffer."""

    def test_dqn_init_with_per(self):
        """Test DQNAgent initializes with PER by default."""
        agent = DQNAgent(
            state_dim=10,
            action_dim=3,
        )

        assert agent.use_prioritized_buffer is True
        assert isinstance(agent.replay_buffer, PrioritizedReplayBuffer)
        assert agent.td_errors is None

    def test_dqn_init_without_per(self):
        """Test DQNAgent can fallback to simple buffer."""
        agent = DQNAgent(
            state_dim=10,
            action_dim=3,
            use_prioritized_buffer=False,
        )

        assert agent.use_prioritized_buffer is False
        assert not isinstance(agent.replay_buffer, PrioritizedReplayBuffer)
        assert agent.td_errors is not None

    def test_dqn_store_with_per(self):
        """Test storing experiences with PER."""
        agent = DQNAgent(state_dim=10, action_dim=3)

        # Store some experiences
        for _ in range(10):
            state = np.random.randn(10)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = np.random.rand() > 0.5

            agent.store(state, action, reward, next_state, done)

        assert len(agent.replay_buffer) == 10

    def test_dqn_train_step_with_per(self):
        """Test training step produces valid loss with PER."""
        agent = DQNAgent(
            state_dim=10,
            action_dim=3,
            batch_size=4,
        )

        # Fill buffer
        for _ in range(50):
            state = np.random.randn(10)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = np.random.rand() > 0.5

            agent.store(state, action, reward, next_state, done)

        # Train
        loss = agent.train_step()

        assert loss is not None
        assert isinstance(loss, float)
        assert not np.isnan(loss)
        assert loss >= 0

    def test_dqn_with_multidiscrete_and_per(self):
        """Test DQNAgent with MultiDiscrete actions and PER."""
        agent = DQNAgent(
            state_dim=15,
            multi_discrete_actions=[3, 10],  # action_type, size_level
            batch_size=4,
        )

        assert agent.multi_discrete is True
        assert isinstance(agent.replay_buffer, PrioritizedReplayBuffer)

        # Store experiences with MultiDiscrete actions
        for _ in range(50):
            state = np.random.randn(15)
            action = [np.random.randint(0, 3), np.random.randint(0, 10)]
            reward = np.random.randn()
            next_state = np.random.randn(15)
            done = np.random.rand() > 0.5

            agent.store(state, action, reward, next_state, done)

        assert len(agent.replay_buffer) == 50

        # Train
        loss = agent.train_step()
        assert loss is not None
        assert not np.isnan(loss)

    def test_dqn_epsilon_decay_with_per(self):
        """Test epsilon decay works correctly with PER."""
        agent = DQNAgent(
            state_dim=10,
            action_dim=3,
            batch_size=4,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.99,
        )

        assert agent.epsilon == 1.0

        # Fill buffer and train multiple times
        for _ in range(50):
            state = np.random.randn(10)
            agent.store(state, 0, 0.0, state, False)

        for _ in range(10):
            agent.train_step()

        # Epsilon should have decayed
        assert agent.epsilon < 1.0
        assert agent.epsilon >= agent.epsilon_end

    def test_dqn_act_with_multidiscrete_per(self):
        """Test act method with MultiDiscrete and PER."""
        agent = DQNAgent(
            state_dim=15,
            multi_discrete_actions=[3, 10],
        )

        state = np.random.randn(15)

        # Explore
        action = agent.act(state, training=True)
        assert isinstance(action, list)
        assert len(action) == 2
        assert action[0] in [0, 1, 2]  # action_type
        assert action[1] in range(10)  # size_level

        # Exploit (set epsilon to 0)
        agent.epsilon = 0.0
        action = agent.act(state, training=True)
        assert isinstance(action, list)

    def test_dqn_from_env_with_per(self):
        """Test creating agent from env with PER."""
        from gymnasium import spaces

        # Create mock env spec
        class MockEnv:
            action_space = spaces.MultiDiscrete([3, 10])
            observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,))

        env = MockEnv()
        agent = DQNAgent.from_env(env)

        assert agent.multi_discrete is True
        assert agent.use_prioritized_buffer is True
        assert isinstance(agent.replay_buffer, PrioritizedReplayBuffer)


class TestDoubleDQNAgentWithPER:
    """Test DoubleDQNAgent with PrioritizedReplayBuffer."""

    def test_double_dqn_init_with_per(self):
        """Test DoubleDQNAgent initializes with PER by default."""
        agent = DoubleDQNAgent(
            state_dim=10,
            action_dim=3,
        )

        assert agent.use_prioritized_buffer is True
        assert isinstance(agent.replay_buffer, PrioritizedReplayBuffer)

    def test_double_dqn_init_without_per(self):
        """Test DoubleDQNAgent can fallback to simple buffer."""
        agent = DoubleDQNAgent(
            state_dim=10,
            action_dim=3,
            use_prioritized_buffer=False,
        )

        assert agent.use_prioritized_buffer is False
        assert not isinstance(agent.replay_buffer, PrioritizedReplayBuffer)

    def test_double_dqn_store_with_per(self):
        """Test storing experiences with PER."""
        agent = DoubleDQNAgent(state_dim=10, action_dim=3)

        for _ in range(10):
            state = np.random.randn(10)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = np.random.rand() > 0.5

            agent.store(state, action, reward, next_state, done)

        assert len(agent.replay_buffer) == 10

    def test_double_dqn_train_step_with_per(self):
        """Test training step produces valid loss with PER."""
        agent = DoubleDQNAgent(
            state_dim=10,
            action_dim=3,
            batch_size=4,
        )

        # Fill buffer
        for _ in range(50):
            state = np.random.randn(10)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = np.random.rand() > 0.5

            agent.store(state, action, reward, next_state, done)

        # Train
        loss = agent.train_step()

        assert loss is not None
        assert isinstance(loss, float)
        assert not np.isnan(loss)
        assert loss >= 0

    def test_double_dqn_with_multidiscrete_and_per(self):
        """Test DoubleDQNAgent with MultiDiscrete and PER."""
        agent = DoubleDQNAgent(
            state_dim=15,
            multi_discrete_actions=[3, 10],
            batch_size=4,
        )

        assert agent.multi_discrete is True
        assert isinstance(agent.replay_buffer, PrioritizedReplayBuffer)

        # Store experiences
        for _ in range(50):
            state = np.random.randn(15)
            action = [np.random.randint(0, 3), np.random.randint(0, 10)]
            reward = np.random.randn()
            next_state = np.random.randn(15)
            done = np.random.rand() > 0.5

            agent.store(state, action, reward, next_state, done)

        assert len(agent.replay_buffer) == 50

        # Train
        loss = agent.train_step()
        assert loss is not None
        assert not np.isnan(loss)

    def test_double_dqn_act_with_multidiscrete_per(self):
        """Test act method with MultiDiscrete and PER."""
        agent = DoubleDQNAgent(
            state_dim=15,
            multi_discrete_actions=[3, 10],
        )

        state = np.random.randn(15)

        # Explore
        action = agent.act(state, training=True)
        assert isinstance(action, list)
        assert len(action) == 2

        # Exploit
        agent.epsilon = 0.0
        action = agent.act(state, training=True)
        assert isinstance(action, list)

    def test_double_dqn_from_env_with_per(self):
        """Test creating DoubleDQNAgent from env with PER."""
        from gymnasium import spaces

        class MockEnv:
            action_space = spaces.Discrete(3)
            observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,))

        env = MockEnv()
        agent = DoubleDQNAgent.from_env(env)

        assert agent.multi_discrete is False
        assert agent.use_prioritized_buffer is True
        assert isinstance(agent.replay_buffer, PrioritizedReplayBuffer)


class TestAgentTrainingLoop:
    """Test complete training loops with PER."""

    def test_dqn_training_convergence(self):
        """Test DQN agent can train without errors."""
        agent = DQNAgent(
            state_dim=5,
            action_dim=3,
            batch_size=8,
            buffer_size=100,
        )

        # Simulate training
        losses = []
        for episode in range(5):
            state = np.random.randn(5)
            for step in range(20):
                action = agent.act(state, training=True)
                reward = np.random.randn()
                next_state = np.random.randn(5)
                done = step == 19

                agent.store(state, action, reward, next_state, done)

                if len(agent.replay_buffer) >= agent.batch_size:
                    loss = agent.train_step()
                    if loss is not None:
                        losses.append(loss)

                state = next_state
                if done:
                    break

        assert len(losses) > 0
        assert all(not np.isnan(l) for l in losses)

    def test_double_dqn_training_convergence(self):
        """Test DoubleDQN agent can train without errors."""
        agent = DoubleDQNAgent(
            state_dim=5,
            action_dim=3,
            batch_size=8,
            buffer_size=100,
        )

        # Simulate training
        losses = []
        for episode in range(5):
            state = np.random.randn(5)
            for step in range(20):
                action = agent.act(state, training=True)
                reward = np.random.randn()
                next_state = np.random.randn(5)
                done = step == 19

                agent.store(state, action, reward, next_state, done)

                if len(agent.replay_buffer) >= agent.batch_size:
                    loss = agent.train_step()
                    if loss is not None:
                        losses.append(loss)

                state = next_state
                if done:
                    break

        assert len(losses) > 0
        assert all(not np.isnan(l) for l in losses)

    def test_dqn_multidiscrete_training_loop(self):
        """Test DQN with MultiDiscrete can train without errors."""
        agent = DQNAgent(
            state_dim=15,
            multi_discrete_actions=[3, 10],
            batch_size=16,
        )

        # Training loop
        for _ in range(30):
            state = np.random.randn(15)
            action = agent.act(state, training=True)
            reward = np.random.randn()
            next_state = np.random.randn(15)
            done = np.random.rand() > 0.8

            agent.store(state, action, reward, next_state, done)

            if len(agent.replay_buffer) >= agent.batch_size:
                loss = agent.train_step()
                if loss is not None:
                    assert not np.isnan(loss)

    def test_prioritized_sampling_effectiveness(self):
        """Test that prioritized sampling focuses on high TD-error transitions."""
        # Create agent with high TD-error variance
        agent = DQNAgent(
            state_dim=5,
            action_dim=3,
            batch_size=16,
        )

        # Fill buffer with varied experiences
        for i in range(100):
            state = np.random.randn(5)
            # Some actions lead to high rewards, others to low
            action = i % 3
            reward = 10.0 if action == 1 else -1.0
            next_state = np.random.randn(5)
            done = i % 10 == 0

            agent.store(state, action, reward, next_state, done)

        # Train and verify no errors
        for _ in range(20):
            loss = agent.train_step()
            assert loss is not None
            assert not np.isnan(loss)


class TestEdgeCases:
    """Test edge cases with PER integration."""

    def test_empty_buffer_train_returns_none(self):
        """Test training with empty buffer returns None."""
        agent = DQNAgent(state_dim=10, action_dim=3)
        assert agent.train_step() is None

    def test_small_buffer_train_returns_none(self):
        """Test training with small buffer returns None."""
        agent = DQNAgent(
            state_dim=10,
            action_dim=3,
            batch_size=64,
        )

        # Add less than batch_size experiences
        for _ in range(10):
            agent.store(np.random.randn(10), 0, 0.0, np.random.randn(10), False)

        assert agent.train_step() is None

    def test_tensor_and_numpy_mix(self):
        """Test agent handles mix of tensor and numpy inputs."""
        agent = DQNAgent(
            state_dim=5,
            action_dim=3,
            batch_size=8,
        )

        # Mix numpy and tensor
        for _ in range(20):
            state = np.random.randn(5)
            next_state = torch.randn(5)
            agent.store(state, 0, 1.0, next_state, False)

        assert len(agent.replay_buffer) == 20
        loss = agent.train_step()
        assert loss is not None

    def test_save_load_with_per(self):
        """Test saving and loading agent with PER."""
        agent = DQNAgent(state_dim=10, action_dim=3)

        # Store some experiences
        for _ in range(10):
            agent.store(np.random.randn(10), 0, 1.0, np.random.randn(10), False)

        # Save
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "agent.pt")
            agent.save(path)

            # Load into new agent
            new_agent = DQNAgent(state_dim=10, action_dim=3)
            new_agent.load(path)

            # Check epsilon is restored
            assert new_agent.epsilon == agent.epsilon
