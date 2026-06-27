import numpy as np
import pandas as pd
import pytest
from custom_envs.trading_env_v6 import TradingEnvV6

def create_mock_data():
    """Creates a simple dataframe for testing."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="4h")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": [100.0] * 10,
        "high": [105.0] * 10,
        "low": [95.0] * 10,
        "close": [100.0] * 10,
        "volume": [1000.0] * 10,
        "gk_volatility": [0.01] * 10,
        "fundingRate": [0.0001] * 10,
        "state_vector": [[0.1, 0.2] for _ in range(10)]
    })
    return df

def test_atomic_flip_short_to_long():
    """Test that the environment flips from full short to full long in one step."""
    df = create_mock_data()
    
    env = TradingEnvV6(
        df=df,
        initial_deposit=1000.0,
        commission=0.0,
        leverage=1.0,
        base_slippage=0.0,
        volatility_factor=0.0,
        t_max=5
    )
    
    env.reset()
    
    # Step 1: Go 100% Short
    action_short = np.array([-1.0])
    _, _, done, _, info = env.step(action_short)
    
    assert info["short_positions"] == 1
    assert info["long_positions"] == 0
    actual_weight = env._get_current_weight(env._get_current_price(), env.get_portfolio_value(env._get_current_price()))
    assert actual_weight < -0.99 # Should be full short
    
    # Step 2: Atomic flip to 100% Long
    action_long = np.array([1.0])
    _, _, done, _, info = env.step(action_long)
    
    assert info["short_positions"] == 0
    assert info["long_positions"] == 1
    actual_weight = env._get_current_weight(env._get_current_price(), env.get_portfolio_value(env._get_current_price()))
    assert actual_weight > 0.99 # Should be full long

def test_atomic_flip_long_to_short():
    """Test that the environment flips from full long to full short in one step."""
    df = create_mock_data()
    
    env = TradingEnvV6(
        df=df,
        initial_deposit=1000.0,
        commission=0.0,
        leverage=1.0,
        base_slippage=0.0,
        volatility_factor=0.0,
        t_max=5
    )
    
    env.reset()
    
    # Step 1: Go 100% Long
    action_long = np.array([1.0])
    _, _, done, _, info = env.step(action_long)
    
    assert info["long_positions"] == 1
    assert info["short_positions"] == 0
    actual_weight = env._get_current_weight(env._get_current_price(), env.get_portfolio_value(env._get_current_price()))
    assert actual_weight > 0.99 # Should be full long
    
    # Step 2: Atomic flip to 100% Short
    action_short = np.array([-1.0])
    _, _, done, _, info = env.step(action_short)
    
    assert info["long_positions"] == 0
    assert info["short_positions"] == 1
    actual_weight = env._get_current_weight(env._get_current_price(), env.get_portfolio_value(env._get_current_price()))
    assert actual_weight < -0.99 # Should be full short
