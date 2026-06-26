"""
Tests for OOSEvalCallback and portfolio_history/episode_sortino functionality.

Test coverage:
1. portfolio_history recording and clearing in environments
2. episode_sortino calculation
3. Multi-asset DataFrame slicing
4. Environment creation (asset × slice combinations)
5. Metrics logging (2D: by regime and by asset)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
from custom_envs.trading_env_v5 import TradingEnvV5
from custom_envs.trading_env_v6 import TradingEnvV6
from agents.callbacks import OOSEvalCallback


class TestPortfolioHistoryAndSortino:
    """Test portfolio_history recording and episode_sortino calculation."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with required columns."""
        np.random.seed(42)
        size = 500

        # Create datetime index
        dates = pd.date_range(start="2024-01-01", periods=size, freq="4h")

        df = pd.DataFrame({
            'timestamp': dates,
            'close': 50000 + np.random.randn(size).cumsum() * 100,
            'gk_volatility': np.random.uniform(0.001, 0.01, size),
            'fundingRate': np.random.uniform(-0.0001, 0.0001, size),
        })

        # Add state_vector (required by environment)
        features = np.random.randn(size, 15).astype(np.float32)
        df['state_vector'] = list(features)

        return df

    @pytest.fixture
    def env_v5(self, sample_df):
        """Create TradingEnvV5 for testing."""
        return TradingEnvV5(
            df=sample_df,
            initial_deposit=100_000.0,
            commission=0.0005,
            leverage=1.0,
            t_max=200,
            domain_randomization=False,
        )

    @pytest.fixture
    def env_v6(self, sample_df):
        """Create TradingEnvV6 for testing."""
        return TradingEnvV6(
            df=sample_df,
            initial_deposit=100_000.0,
            commission=0.0005,
            leverage=1.0,
            t_max=200,
            domain_randomization=False,
        )

    def test_portfolio_history_initialized(self, env_v5, env_v6):
        """Test that portfolio_history is initialized in both environments."""
        assert hasattr(env_v5, 'portfolio_history')
        assert hasattr(env_v6, 'portfolio_history')
        assert isinstance(env_v5.portfolio_history, list)
        assert isinstance(env_v6.portfolio_history, list)

    def test_portfolio_history_cleared_on_reset(self, env_v5, env_v6):
        """Test that portfolio_history is cleared on reset."""
        # Add some fake history
        env_v5.portfolio_history = [100000, 101000, 102000]
        env_v6.portfolio_history = [100000, 101000, 102000]

        # Reset
        env_v5.reset()
        env_v6.reset()

        # Should be empty
        assert len(env_v5.portfolio_history) == 0
        assert len(env_v6.portfolio_history) == 0

    def test_portfolio_history_records_on_each_step(self, env_v5):
        """Test that portfolio_value is recorded on each step."""
        obs, _ = env_v5.reset()
        initial_len = len(env_v5.portfolio_history)

        # Take a few steps
        for _ in range(10):
            action = env_v5.action_space.sample()
            obs, reward, done, _, info = env_v5.step(action)
            if done:
                break

        # History should have grown
        assert len(env_v5.portfolio_history) > initial_len
        # Each entry should be a float
        assert all(isinstance(x, (int, float)) for x in env_v5.portfolio_history)

    def test_episode_sortino_in_info_on_done(self, env_v5):
        """Test that episode_sortino is returned in info when episode ends."""
        obs, _ = env_v5.reset()

        # Run until done
        done = False
        step_count = 0
        while not done and step_count < 300:
            action = env_v5.action_space.sample()
            obs, reward, done, truncated, info = env_v5.step(action)
            step_count += 1

        # When done, should have episode_sortino
        assert "episode_sortino" in info
        assert isinstance(info["episode_sortino"], (int, float))

    def test_calculate_sortino_from_history(self, env_v5):
        """Test Sortino calculation method."""
        # Create a known portfolio history
        env_v5.portfolio_history = [100000, 101000, 102000, 101000, 103000]

        sortino = env_v5._calculate_sortino_from_history()

        # Should return a finite number
        assert isinstance(sortino, (int, float))
        assert not np.isnan(sortino)
        assert not np.isinf(sortino)

    def test_sortino_with_empty_history(self, env_v5):
        """Test Sortino calculation with empty history."""
        env_v5.portfolio_history = []
        sortino = env_v5._calculate_sortino_from_history()
        assert sortino == 0.0

    def test_sortino_with_single_value(self, env_v5):
        """Test Sortino calculation with single value (no returns)."""
        env_v5.portfolio_history = [100000]
        sortino = env_v5._calculate_sortino_from_history()
        assert sortino == 0.0


class TestOOSEvalCallbackSlicing:
    """Test multi-asset DataFrame slicing in OOSEvalCallback."""

    @pytest.fixture
    def sample_multi_asset_dfs(self):
        """Create sample DataFrames for multiple assets."""
        np.random.seed(42)

        def create_df(start_date):
            size = 1000
            dates = pd.date_range(start=start_date, periods=size, freq="4h")
            df = pd.DataFrame({
                'timestamp': dates,
                'close': 50000 + np.random.randn(size).cumsum() * 100,
                'gk_volatility': np.random.uniform(0.001, 0.01, size),
                'fundingRate': np.random.uniform(-0.0001, 0.0001, size),
            })
            features = np.random.randn(size, 15).astype(np.float32)
            df['state_vector'] = list(features)
            return df

        return {
            "BTCUSDT": create_df("2023-01-01"),
            "ETHUSDT": create_df("2023-01-01"),
            "BNBUSDT": create_df("2023-01-01"),
        }

    @pytest.fixture
    def val_slices(self):
        """Create validation slices."""
        return {
            "bull_1": ("2023-02-01", "2023-03-01"),
            "bear_1": ("2023-04-01", "2023-05-01"),
            "flat_1": ("2023-06-01", "2023-07-01"),
        }

    def test_init_callback_creates_correct_env_count(
        self, sample_multi_asset_dfs, val_slices
    ):
        """Test that correct number of environments is created."""
        # Mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=(np.array([[0.0]]), None))

        callback = OOSEvalCallback(
            val_dfs=sample_multi_asset_dfs,
            val_slices=val_slices,
            eval_freq=1000,
            verbose=0,
        )

        # Mock the model attribute
        callback.model = mock_model

        # Initialize
        callback._init_callback()

        # Should have 3 assets × 3 slices = 9 environments
        assert len(callback.val_envs) == 9

    def test_callback_skips_empty_slices(
        self, sample_multi_asset_dfs, val_slices
    ):
        """Test that empty slices are skipped."""
        # Add a slice with invalid dates
        val_slices_invalid = {
            **val_slices,
            "empty_slice": ("2030-01-01", "2030-02-01"),  # Future dates
        }

        mock_model = Mock()
        mock_model.predict = Mock(return_value=(np.array([[0.0]]), None))

        callback = OOSEvalCallback(
            val_dfs=sample_multi_asset_dfs,
            val_slices=val_slices_invalid,
            eval_freq=1000,
            verbose=0,
        )
        callback.model = mock_model
        callback._init_callback()

        # Should skip empty slice, still have 3 × 3 = 9 environments
        assert len(callback.val_envs) == 9

    def test_callback_envs_have_correct_config(
        self, sample_multi_asset_dfs, val_slices
    ):
        """Test that created environments have deterministic config."""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=(np.array([[0.0]]), None))

        callback = OOSEvalCallback(
            val_dfs=sample_multi_asset_dfs,
            val_slices=val_slices,
            eval_freq=1000,
            verbose=0,
        )
        callback.model = mock_model
        callback._init_callback()

        # Check one environment
        env = list(callback.val_envs.values())[0]

        # Should have deterministic settings
        assert env.domain_randomization == False
        # t_max should be len(df) - 1 when t_max=None is passed
        assert env.t_max == len(env.df) - 1


class TestOOSEvalCallbackMetrics:
    """Test metrics logging in OOSEvalCallback."""

    @pytest.fixture
    def minimal_setup(self):
        """Create minimal setup for callback testing."""
        np.random.seed(42)

        # Create minimal DFs - need enough data for all slices
        def create_df():
            size = 1000  # Increased to cover all validation periods
            dates = pd.date_range(start="2023-01-01", periods=size, freq="4h")
            df = pd.DataFrame({
                'timestamp': dates,
                'close': 50000 + np.random.randn(size).cumsum() * 50,
                'gk_volatility': np.random.uniform(0.001, 0.01, size),
                'fundingRate': np.random.uniform(-0.0001, 0.0001, size),
            })
            features = np.random.randn(size, 15).astype(np.float32)
            df['state_vector'] = list(features)
            return df

        val_dfs = {
            "BTCUSDT": create_df(),
            "ETHUSDT": create_df(),
        }

        val_slices = {
            "bull_1": ("2023-01-05", "2023-01-25"),
            "bear_1": ("2023-02-05", "2023-02-25"),
            "flat_1": ("2023-03-05", "2023-03-25"),
        }

        return val_dfs, val_slices

    def test_evaluate_on_slice_returns_metrics(self, minimal_setup):
        """Test that _evaluate_on_slice returns correct metrics."""
        val_dfs, val_slices = minimal_setup

        # Create one environment
        df = val_dfs["BTCUSDT"].copy()

        # Filter by timestamp column since DF might not have datetime index
        if 'timestamp' in df.columns:
            df_slice = df[(df['timestamp'] >= "2023-01-05") & (df['timestamp'] <= "2023-01-25")].reset_index(drop=True)
        else:
            df_slice = df.loc["2023-01-05":"2023-01-25"].reset_index(drop=True)

        env = TradingEnvV6(df=df_slice, domain_randomization=False, t_max=None)

        # Mock model
        mock_model = Mock()
        mock_model.predict = Mock(side_effect=lambda obs, deterministic: (np.array([[0.0]]), None))

        callback = OOSEvalCallback(
            val_dfs=val_dfs,
            val_slices=val_slices,
            eval_freq=1000,
            verbose=0,
        )
        callback.model = mock_model

        # Evaluate
        result = callback._evaluate_on_slice(env)

        # Should have required metrics
        assert "pnl" in result
        assert "sortino" in result
        assert "num_steps" in result
        assert isinstance(result["pnl"], (int, float))
        assert isinstance(result["sortino"], (int, float))
        assert isinstance(result["num_steps"], int)

    def test_logger_record_called_correctly(self, minimal_setup):
        """Test that logger.record is called with correct metrics."""
        val_dfs, val_slices = minimal_setup

        mock_model = Mock()
        mock_model.predict = Mock(return_value=(np.array([[0.0]]), None))

        callback = OOSEvalCallback(
            val_dfs=val_dfs,
            val_slices=val_slices,
            eval_freq=1,  # Evaluate on first step
            verbose=0,
        )
        callback.model = mock_model

        # Initialize - this sets up the parent callback including logger
        callback._init_callback()

        # Mock the logger.record method
        original_record = callback.logger.record
        callback.logger.record = Mock()

        # Simulate being at eval_freq step
        callback.num_timesteps = 1

        # Run _on_step
        callback._on_step()

        # Verify logger.record was called
        assert callback.logger.record.called

        # Get all logged metric names
        logged_metrics = set()
        for call in callback.logger.record.call_args_list:
            logged_metrics.add(call[0][0])  # First arg is metric name

        # Debug: print what was logged
        if not any("bull" in m for m in logged_metrics):
            print(f"Logged metrics: {sorted(logged_metrics)}")

        # Should have regime metrics
        assert any("bull" in m for m in logged_metrics), f"No 'bull' metrics found in: {sorted(logged_metrics)}"
        assert any("bear" in m for m in logged_metrics), f"No 'bear' metrics found in: {sorted(logged_metrics)}"
        assert any("flat" in m for m in logged_metrics), f"No 'flat' metrics found in: {sorted(logged_metrics)}"

        # Should have asset metrics (BTC, ETH)
        assert any("BTC" in m or "ETH" in m for m in logged_metrics)

        # Should have overall metrics
        assert "val/sortino_overall" in logged_metrics
        assert "val/pnl_total" in logged_metrics

        # Restore original
        callback.logger.record = original_record


class TestOSEvalCallbackEdgeCases:
    """Test edge cases in OOSEvalCallback."""

    @pytest.fixture
    def minimal_setup(self):
        """Create minimal setup for callback testing."""
        np.random.seed(42)

        # Create minimal DFs - need enough data for all slices
        def create_df():
            size = 1000  # Increased to cover all validation periods
            dates = pd.date_range(start="2023-01-01", periods=size, freq="4h")
            df = pd.DataFrame({
                'timestamp': dates,
                'close': 50000 + np.random.randn(size).cumsum() * 50,
                'gk_volatility': np.random.uniform(0.001, 0.01, size),
                'fundingRate': np.random.uniform(-0.0001, 0.0001, size),
            })
            features = np.random.randn(size, 15).astype(np.float32)
            df['state_vector'] = list(features)
            return df

        val_dfs = {
            "BTCUSDT": create_df(),
            "ETHUSDT": create_df(),
        }

        val_slices = {
            "bull_1": ("2023-01-05", "2023-01-25"),
            "bear_1": ("2023-02-05", "2023-02-25"),
            "flat_1": ("2023-03-05", "2023-03-25"),
        }

        return val_dfs, val_slices

    def test_callback_with_datetime_index(self):
        """Test callback works with DatetimeIndex."""
        np.random.seed(42)

        # Create DF with DatetimeIndex
        dates = pd.date_range(start="2023-01-01", periods=500, freq="4h")
        df = pd.DataFrame(
            index=dates,
            data={
                'close': 50000 + np.random.randn(500).cumsum() * 50,
                'gk_volatility': np.random.uniform(0.001, 0.01, 500),
                'fundingRate': np.random.uniform(-0.0001, 0.0001, 500),
            }
        )
        features = np.random.randn(500, 15).astype(np.float32)
        df['state_vector'] = list(features)

        val_dfs = {"BTCUSDT": df}
        val_slices = {"bull_1": ("2023-01-05", "2023-02-01")}

        mock_model = Mock()
        mock_model.predict = Mock(return_value=(np.array([[0.0]]), None))

        callback = OOSEvalCallback(
            val_dfs=val_dfs,
            val_slices=val_slices,
            eval_freq=1000,
            verbose=0,
        )
        callback.model = mock_model

        # Should not raise
        callback._init_callback()
        assert len(callback.val_envs) == 1

    def test_callback_with_timestamp_column(self):
        """Test callback works with timestamp column."""
        np.random.seed(42)

        # Create DF with timestamp column
        dates = pd.date_range(start="2023-01-01", periods=500, freq="4h")
        df = pd.DataFrame({
            'timestamp': dates,
            'close': 50000 + np.random.randn(500).cumsum() * 50,
            'gk_volatility': np.random.uniform(0.001, 0.01, 500),
            'fundingRate': np.random.uniform(-0.0001, 0.0001, 500),
        })
        features = np.random.randn(500, 15).astype(np.float32)
        df['state_vector'] = list(features)

        val_dfs = {"BTCUSDT": df}
        val_slices = {"bull_1": ("2023-01-05", "2023-02-01")}

        mock_model = Mock()
        mock_model.predict = Mock(return_value=(np.array([[0.0]]), None))

        callback = OOSEvalCallback(
            val_dfs=val_dfs,
            val_slices=val_slices,
            eval_freq=1000,
            verbose=0,
        )
        callback.model = mock_model

        # Should not raise
        callback._init_callback()
        assert len(callback.val_envs) == 1

    def test_callback_raises_without_datetime_index_or_timestamp(self):
        """Test that callback raises error without DatetimeIndex or timestamp column."""
        np.random.seed(42)

        # Create DF without datetime index or timestamp column
        df = pd.DataFrame({
            'close': 50000 + np.random.randn(500).cumsum() * 50,
            'gk_volatility': np.random.uniform(0.001, 0.01, 500),
        })
        features = np.random.randn(500, 15).astype(np.float32)
        df['state_vector'] = list(features)

        val_dfs = {"BTCUSDT": df}
        val_slices = {"bull_1": ("2023-01-05", "2023-02-01")}

        mock_model = Mock()
        mock_model.predict = Mock(return_value=(np.array([[0.0]]), None))

        callback = OOSEvalCallback(
            val_dfs=val_dfs,
            val_slices=val_slices,
            eval_freq=1000,
            verbose=0,
        )
        callback.model = mock_model

        # Should raise ValueError
        with pytest.raises(ValueError, match="must have DatetimeIndex or 'timestamp' column"):
            callback._init_callback()

    def test_best_model_saving(self, minimal_setup, tmp_path):
        """Test that best model is saved correctly."""
        from unittest.mock import patch

        val_dfs, val_slices = minimal_setup

        mock_model = Mock()
        mock_model.predict = Mock(return_value=(np.array([[0.0]]), None))

        save_path = tmp_path / "best_model.zip"

        callback = OOSEvalCallback(
            val_dfs=val_dfs,
            val_slices=val_slices,
            eval_freq=1,
            best_model_save_path=str(save_path),
            verbose=0,
        )
        callback.model = mock_model

        # Initialize
        callback._init_callback()

        # First eval: should save (best_sortino starts at -inf)
        callback.num_timesteps = 1
        callback._on_step()

        # Verify model.save was called
        mock_model.save.assert_called_once()

    def test_on_step_returns_true(self, minimal_setup):
        """Test that _on_step always returns True (doesn't stop training)."""
        val_dfs, val_slices = minimal_setup

        mock_model = Mock()
        mock_model.predict = Mock(return_value=(np.array([[0.0]]), None))

        callback = OOSEvalCallback(
            val_dfs=val_dfs,
            val_slices=val_slices,
            eval_freq=1000,
            verbose=0,
        )
        callback.model = mock_model
        callback._init_callback()

        # Should return True even when not evaluating
        callback.num_timesteps = 500
        result = callback._on_step()
        assert result is True

        # Should return True when evaluating
        callback.num_timesteps = 1000
        result = callback._on_step()
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
