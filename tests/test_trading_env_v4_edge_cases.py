"""
Unit tests for TradingEnvV4 edge cases.

Critical scenarios to test:
1. Deposit=0 with open positions → can still close
2. Full cycle: buy → sell → buy → sell
3. get_possible_actions() actions are actually executable
4. Cannot open new positions when deposit is low
5. Short positions work correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from custom_envs.trading_env_v4 import TradingEnvV4


class TestTradingEnvV4EdgeCases:
    """Test edge cases that caused critical bugs."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        np.random.seed(42)
        features = np.random.randn(1000, 15).astype(np.float32)
        prices = 50000 + np.random.randn(1000).cumsum() * 10

        return TradingEnvV4(
            features=features,
            real_prices=prices,
            initial_deposit=100_000,
            t_max=500,
        )

    def test_can_close_position_when_deposit_is_zero(self, env):
        """CRITICAL: When deposit=0 with open long, MUST be able to close.

        This was the main bug: agent buys 100%, deposit becomes 0,
        then cannot close position → stuck forever.
        """
        # Reset
        obs, _ = env.reset()
        assert env.deposit == 100_000

        # Buy 100%
        action = [1, 9]  # BUY 100%
        obs, reward, done, _, info = env.step(action)

        # Verify: deposit should be 0, we have long position
        assert env.deposit == 0, f"Deposit should be 0, got {env.deposit}"
        assert len(env.positions_long) > 0, "Should have long position"
        assert info["executed"] == True, "BUY should execute"

        # CRITICAL TEST: Can we close the position?
        possible = env.get_possible_actions()
        assert len(possible) > 1, "Should have more than just HOLD when have position"

        # Should have SELL actions (to close long)
        sell_actions = [a for a in possible if a[0] == 2]
        assert len(sell_actions) > 0, "Should have SELL actions to close long"

        # Execute SELL 100%
        action = [2, 9]  # SELL 100%
        obs, reward, done, _, info = env.step(action)

        # CRITICAL VERIFY: Position should be closed, deposit restored
        assert len(env.positions_long) == 0, "Long position should be closed"
        assert env.deposit > 50_000, f"Deposit should be restored, got {env.deposit}"
        assert info["executed"] == True, "SELL should execute"

    def test_full_trading_cycle(self, env):
        """Test full cycle: buy → sell → buy → sell.

        Agent should be able to trade repeatedly without getting stuck.
        """
        obs, _ = env.reset()

        for cycle in range(3):
            # Buy 100%
            action = [1, 9]
            obs, reward, done, _, info = env.step(action)
            assert info["executed"], f"Cycle {cycle}: BUY should execute"
            assert len(env.positions_long) > 0, f"Cycle {cycle}: Should have long"

            # Sell 100%
            action = [2, 9]
            obs, reward, done, _, info = env.step(action)
            assert info["executed"], f"Cycle {cycle}: SELL should execute"
            assert len(env.positions_long) == 0, f"Cycle {cycle}: Long should be closed"
            assert env.deposit > 50_000, f"Cycle {cycle}: Deposit should be restored"

    def test_possible_actions_are_executable(self, env):
        """CRITICAL: Every action from get_possible_actions() MUST execute.

        This was the bug: get_possible_actions() returned SELL,
        but _execute_sell() returned False due to _can_trade() check.
        """
        obs, _ = env.reset()

        # Buy 100% to get deposit=0 with open long
        action = [1, 9]
        obs, reward, done, _, info = env.step(action)
        assert env.deposit == 0, "Deposit should be 0"
        assert len(env.positions_long) > 0, "Should have long position"

        # Get possible actions
        possible = env.get_possible_actions()

        # Should have SELL actions to close the long
        sell_actions = [a for a in possible if a[0] == 2]
        assert len(sell_actions) > 0, "Should have SELL actions"

        # Execute SELL 100% to verify it actually executes
        action = [2, 9]
        prev_long_count = len(env.positions_long)
        obs, reward, done, _, info = env.step(action)

        # Verify: position was closed
        assert len(env.positions_long) < prev_long_count, \
            "SELL should close long position"
        assert env.deposit > 50_000, \
            "Deposit should be restored after closing position"

        # Test the other direction: sell to open short
        env.reset()
        action = [2, 9]  # Open short
        obs, reward, done, _, info = env.step(action)
        assert env.deposit == 0, "Deposit should be 0 after opening short"
        assert len(env.positions_short) > 0, "Should have short position"

        # Should be able to close with BUY
        possible = env.get_possible_actions()
        buy_actions = [a for a in possible if a[0] == 1]
        assert len(buy_actions) > 0, "Should have BUY actions to close short"

        # Execute BUY
        action = [1, 9]
        prev_short_count = len(env.positions_short)
        obs, reward, done, _, info = env.step(action)

        assert len(env.positions_short) < prev_short_count, \
            "BUY should close short position"
        assert env.deposit > 50_000, \
            "Deposit should be restored after closing short"

    def test_cannot_open_new_positions_when_deposit_is_low(self, env):
        """When deposit is low and no positions, cannot open new ones.

        This is correct behavior: need money to open positions.
        """
        obs, _ = env.reset()

        # Set deposit to very low value
        env.deposit = 100  # Way below 1000 threshold

        # Get possible actions
        possible = env.get_possible_actions()

        # Should only have HOLD
        assert len(possible) == 1, "Should only have HOLD when no money and no positions"
        assert possible[0] == [0, 0], "Only action should be HOLD"

    def test_short_position_cycle(self, env):
        """Test short positions: sell → buy → sell → buy."""
        obs, _ = env.reset()

        # Sell 100% (open short)
        action = [2, 9]
        obs, reward, done, _, info = env.step(action)
        assert info["executed"], "SELL (open short) should execute"
        assert len(env.positions_short) > 0, "Should have short position"
        initial_deposit = env.deposit

        # Buy 100% (close short)
        action = [1, 9]
        obs, reward, done, _, info = env.step(action)
        assert info["executed"], "BUY (close short) should execute"
        assert len(env.positions_short) == 0, "Short should be closed"
        assert env.deposit > initial_deposit - 10_000, "Deposit should be restored"

    def test_can_close_short_when_deposit_is_zero(self, env):
        """When deposit=0 with open short, MUST be able to close."""
        obs, _ = env.reset()

        # Sell 100% (open short) - deposit will go to 0
        action = [2, 9]
        obs, reward, done, _, info = env.step(action)
        assert env.deposit == 0, "Deposit should be 0"
        assert len(env.positions_short) > 0, "Should have short"

        # Should be able to close with BUY
        possible = env.get_possible_actions()
        buy_actions = [a for a in possible if a[0] == 1]
        assert len(buy_actions) > 0, "Should have BUY actions to close short"

        # Execute BUY
        action = [1, 9]
        obs, reward, done, _, info = env.step(action)
        assert len(env.positions_short) == 0, "Short should be closed"
        assert env.deposit > 50_000, "Deposit should be restored"

    def test_partial_position_close(self, env):
        """Test closing partial position (50%)."""
        obs, _ = env.reset()

        # Buy 100%
        action = [1, 9]
        obs, reward, done, _, info = env.step(action)
        initial_volume = sum(v for _, v in env.positions_long)

        # Close 50% (size_idx 4 = 50%)
        action = [2, 4]
        obs, reward, done, _, info = env.step(action)

        current_volume = sum(v for _, v in env.positions_long)
        expected_volume = initial_volume * 0.5

        assert abs(current_volume - expected_volume) < 0.001, \
            f"Should close 50%, expected {expected_volume}, got {current_volume}"
        assert env.deposit > 40_000, "Should have partial deposit back"

    def test_random_agent_doesnt_get_stuck(self, env):
        """Random agent should not get stuck in HOLD loop."""
        np.random.seed(42)
        obs, _ = env.reset()

        actions_log = []
        for i in range(200):
            possible = env.get_possible_actions()
            action = possible[np.random.randint(len(possible))]
            actions_log.append(action)

            obs, reward, done, _, info = env.step(action)
            if done:
                break

        # Count action types
        hold_count = sum(1 for a in actions_log if a[0] == 0)
        buy_count = sum(1 for a in actions_log if a[0] == 1)
        sell_count = sum(1 for a in actions_log if a[0] == 2)

        total = hold_count + buy_count + sell_count

        # Should have diverse actions, not just HOLD
        assert buy_count / total > 0.3, f"Too few BUY: {buy_count/total:.2%}"
        assert sell_count / total > 0.3, f"Too few SELL: {sell_count/total:.2%}"
        assert hold_count / total < 0.2, f"Too many HOLD: {hold_count/total:.2%}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
