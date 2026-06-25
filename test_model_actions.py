import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from custom_envs.trading_env_v5 import TradingEnvV5
from core.data.data_loader import load_crypto_data
from core.features.feature_generator import FeatureGenerator

df = load_crypto_data(symbol='BTCUSDT', start_date='2022-01-01', interval='4h', source='bybit_futures')
fg = FeatureGenerator()
data_features = fg.transform(df)

TEST_SIZE = 360
test_df = data_features.iloc[-TEST_SIZE:].reset_index(drop=True)

test_env = TradingEnvV5(df=test_df, continuous_action=True, t_max=len(test_df), initial_deposit=100_000.0)
vec_env = DummyVecEnv([lambda: test_env])

N_STACK = 10
vec_env = VecFrameStack(vec_env, n_stack=N_STACK)
vec_env = VecNormalize.load("models/vec_normalize_prod.pkl", vec_env)
vec_env.training = False
vec_env.norm_reward = False

model = PPO.load("models/ppo_prod_model")

obs = vec_env.reset()
print("First 10 actions:")
for i in range(10):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = vec_env.step(action)
    print(f"Step {i}: Raw action: {action}")
