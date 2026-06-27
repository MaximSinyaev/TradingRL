import os
import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

sys.path.append('.')

from core.data.data_loader import load_crypto_data
from core.data.data_splitter import create_purged_train_val_split
from core.features.feature_generator import FeatureGenerator
from core.config import VAL_SLICES, N_STACK
from custom_envs.trading_env_v6 import TradingEnvV6
from core.experiment_manager import get_experiment_paths

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from agents.ppo_agent import create_ppo_agent
from agents.callbacks import OOSEvalCallback, TradingMetricsCallback
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.monitor import Monitor

def make_env(df_list, t_max=1440):
    env = TradingEnvV6(
        df=df_list,
        total_assets=3, 
        initial_deposit=100000.0,
        commission=0.0005,
        leverage=1.0,
        t_max=t_max
    )
    return Monitor(env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2000000)
    parser.add_argument("--n_stack", type=int, default=N_STACK)
    parser.add_argument("--oos_freq", type=int, default=100000)
    parser.add_argument("--extractor", type=str, default="transformer", choices=["mlp", "cnn", "gru", "transformer"])
    parser.add_argument("--exp_name", type=str, default="ppo_purged_v6_multi", help="Base experiment name. Extractor will be appended automatically unless you change the code.")
    args = parser.parse_args()
    
    exp_name = f"{args.exp_name}_{args.extractor}"
    model_path, norm_path, tb_log_dir = get_experiment_paths(exp_name)
    
    # 1. Download Data and Generate Features FIRST
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    processed_dfs_dict = {}
    
    # Check or train HMM model automatically
    from core.features.hmm_helper import get_or_train_hmm
    hmm_path = get_or_train_hmm()
    
    fg = FeatureGenerator(hmm_path=hmm_path)

    for i, sym in enumerate(symbols):
        print(f"Loading and processing {sym}...")
        df = load_crypto_data(symbol=sym, start_date="2020-01-01", end_date="2026-06-26", interval="4h", use_cache=True)
        processed = fg.transform(df)
        processed['asset_idx'] = i
        processed_dfs_dict[sym] = processed

    # 2. Create Purged Split on ALREADY PROCESSED data
    # Embargo must be equal to the max lookback of our features to prevent leakage (42 as agreed)
    train_dfs_chunks, val_dfs_dict = create_purged_train_val_split(
        dfs_dict=processed_dfs_dict,
        val_slices=VAL_SLICES,
        embargo_candles=42
    )

    # Filter out empty or too short chunks just in case
    processed_train_dfs = [df for df in train_dfs_chunks if not df.empty and len(df) > 50]

    # 3. Setup Validation Environments
    eval_envs_dict = {}
    for asset_name, slices_dict in val_dfs_dict.items():
        for slice_name, df_slice in slices_dict.items():
            if df_slice.empty or len(df_slice) < 10:
                continue
            
            env = DummyVecEnv([lambda df=df_slice: TradingEnvV6(df=df, total_assets=3, domain_randomization=False, t_max=None)])
            env = VecFrameStack(env, n_stack=args.n_stack)
            env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
            env.training = False # DO NOT update moving averages during validation!
            eval_envs_dict[(asset_name, slice_name)] = env

    print(f"Created {len(eval_envs_dict)} validation environments.")

    # 4. Setup Training Environment
    vec_env = DummyVecEnv([lambda: make_env(processed_train_dfs, t_max=1440)])
    vec_env = VecFrameStack(vec_env, n_stack=args.n_stack)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 5. Callbacks
    run = wandb.init(
        project="trading_rl",
        name=exp_name,
        sync_tensorboard=True,
        monitor_gym=True,
    )

    oos_callback = OOSEvalCallback(
        eval_envs_dict=eval_envs_dict,
        eval_freq=args.oos_freq,
        experiment_name=exp_name,
        best_model_save_path=model_path,
        verbose=1
    )

    metrics_callback = TradingMetricsCallback()
    wandb_callback = WandbCallback(gradient_save_freq=1000, model_save_path=model_path, verbose=2)

    # 6. Initialize Model
    hmm_cols = [c for c in processed_train_dfs[0].columns if 'hmm_regime' in c]
    
    model = create_ppo_agent(
        vec_env, 
        extractor_type=args.extractor, 
        num_hmm_states=len(hmm_cols), 
        n_stack=args.n_stack,
        tensorboard_log=tb_log_dir
    )

    print(f"Start learning on {len(processed_train_dfs)} chunks for {args.timesteps} timesteps...")
    
    # 7. Train
    model.learn(total_timesteps=args.timesteps, callback=[metrics_callback, oos_callback, wandb_callback], progress_bar=True)

    # 8. Save
    model.save(model_path)
    vec_env.save(norm_path)
    wandb.finish()
    
    print("Training complete and models saved.")

if __name__ == "__main__":
    main()
