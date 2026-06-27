import numpy as np
import pandas as pd
from custom_envs.trading_env_v6 import TradingEnvV6

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

env = TradingEnvV6(df=df, initial_deposit=1000.0, commission=0.0, leverage=1.0, base_slippage=0.0, volatility_factor=0.0, t_max=5)
env.reset()

print("STEP 1: Long")
_, _, _, _, info = env.step(np.array([1.0]))
print(f"Info after Long: {info['long_positions']=}, {info['short_positions']=}")
print(f"Positions: Long={env.positions_long}, Short={env.positions_short}")
print(f"Deposit={env.deposit}")

print("\nSTEP 2: Short")
_, _, _, _, info = env.step(np.array([-1.0]))
print(f"Info after Short: {info['long_positions']=}, {info['short_positions']=}")
print(f"Positions: Long={env.positions_long}, Short={env.positions_short}")
print(f"Deposit={env.deposit}")

