import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Training PPO with Purged Cross-Validation\nThis notebook trains a PPO agent on multi-asset data using Purged Cross-Validation to prevent data leakage from overlapping indicators."),
    nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append('..')

from core.data.data_loader import load_crypto_data
from core.data.data_splitter import create_purged_train_val_split
from core.features.feature_generator import FeatureGenerator
from core.config import VAL_SLICES
from custom_envs.trading_env_v6 import TradingEnvV6
from core.experiment_manager import get_experiment_paths

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from agents.callbacks import OOSEvalCallback, TradingMetricsCallback"""),

    nbf.v4.new_markdown_cell("## 1. Load and Split Data\nWe load the data and punch holes for the validation slices, leaving an embargo of 42 candles around each hole."),
    nbf.v4.new_code_cell("""# 1. Download Data
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
dfs_dict = {}

for sym in symbols:
    print(f"Loading {sym}...")
    df = load_crypto_data(symbol=sym, start_date="2020-01-01", end_date="2024-12-31", interval="4h")
    dfs_dict[sym] = df

# 2. Create Purged Split
train_dfs_chunks, val_dfs_dict = create_purged_train_val_split(
    dfs_dict=dfs_dict,
    val_slices=VAL_SLICES,
    embargo_candles=42
)
"""),

    nbf.v4.new_markdown_cell("## 2. Feature Generation\nGenerate features for all train chunks and validation slices."),
    nbf.v4.new_code_cell("""fg = FeatureGenerator()

print("Generating features for Train chunks...")
processed_train_dfs = []
for i, chunk in enumerate(train_dfs_chunks):
    processed = fg.transform(chunk)
    processed_train_dfs.append(processed)
    
print("Generating features for Validation slices...")
processed_val_dict = {}
for sym, slices in val_dfs_dict.items():
    processed_val_dict[sym] = {}
    for slice_name, val_df in slices.items():
        processed_val_dict[sym][slice_name] = fg.transform(val_df)
"""),

    nbf.v4.new_markdown_cell("## 3. Environment Setup\nWe use TradingEnvV6, passing the list of disjoint training chunks."),
    nbf.v4.new_code_cell("""def make_env():
    # Pass the list of all processed chunks. The environment will randomly sample a chunk on reset.
    env = TradingEnvV6(
        df=processed_train_dfs, 
        initial_deposit=100000.0,
        commission=0.0005,
        leverage=1.0,
        t_max=1440 # Roughly 6 months
    )
    return env

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
"""),

    nbf.v4.new_markdown_cell("## 4. Callbacks and Training"),
    nbf.v4.new_code_cell("""exp_name = "ppo_purged_v6_multi"
model_path, norm_path, tb_log_dir = get_experiment_paths(exp_name)

# Using the first symbol's val dict for the callback to keep it simple, or iterate
oos_callback = OOSEvalCallback(
    eval_envs_dict={name: DummyVecEnv([lambda: TradingEnvV6(df=df)]) for name, df in processed_val_dict["BTCUSDT"].items()},
    eval_freq=10000,
    log_dir=tb_log_dir
)

metrics_callback = TradingMetricsCallback(log_freq=1000)

model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01,
    tensorboard_log=tb_log_dir,
    verbose=1
)

model.learn(total_timesteps=100000, callback=[metrics_callback, oos_callback])

# Save
model.save(model_path)
vec_env.save(norm_path)
"""),

    nbf.v4.new_markdown_cell("## 5. Final Validation (Random Train Chunk vs OOS Slice)\nWe test the trained model on a random chunk from the training set, and then on a validation slice."),
    nbf.v4.new_code_cell("""def evaluate_on_env(env, model, name):
    print(f"\\n--- Evaluating on {name} ---")
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
    final_info = info[0] if isinstance(info, list) else info
    print(f"Final PnL: {final_info.get('pnl', 0):.2f}%")
    print(f"Max Drawdown: {final_info.get('drawdown', 0):.2f}%")
    print(f"Sortino Proxy: {final_info.get('episode_sortino', 0):.4f}")

# 1. Random Train Chunk
import random
random_train_idx = random.randint(0, len(processed_train_dfs)-1)
eval_train_env = DummyVecEnv([lambda: TradingEnvV6(df=processed_train_dfs[random_train_idx], t_max=len(processed_train_dfs[random_train_idx]))])
eval_train_env = VecNormalize.load(norm_path, eval_train_env)
eval_train_env.training = False
eval_train_env.norm_reward = False

evaluate_on_env(eval_train_env, model, f"Train Chunk {random_train_idx}")

# 2. Random Val Slice
val_slice_name = "bear_1"
eval_val_env = DummyVecEnv([lambda: TradingEnvV6(df=processed_val_dict["BTCUSDT"][val_slice_name], t_max=len(processed_val_dict["BTCUSDT"][val_slice_name]))])
eval_val_env = VecNormalize.load(norm_path, eval_val_env)
eval_val_env.training = False
eval_val_env.norm_reward = False

evaluate_on_env(eval_val_env, model, f"OOS Val Slice '{val_slice_name}'")
""")
]

nb['cells'] = cells
with open('notebooks/13. train_ppo_purged.ipynb', 'w') as f:
    nbf.write(nb, f)
