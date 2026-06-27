import time
import psutil
import torch
import platform
import warnings
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from core.data.data_loader import load_crypto_data
from core.features.feature_generator import FeatureGenerator
from core.features.hmm_helper import get_or_train_hmm
from custom_envs.trading_env_v6 import TradingEnvV6
from agents.extractors import GatedTransformerExtractor

warnings.filterwarnings("ignore")

N_STACK = 10
TIMESTEPS = 10000

def get_system_info():
    print("=" * 50)
    print("🖥️  SYSTEM HARDWARE INFO")
    print("=" * 50)
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"CPU: {platform.processor()}")
    print(f"Logical CPU Cores: {psutil.cpu_count(logical=True)}")
    print(f"Physical CPU Cores: {psutil.cpu_count(logical=False)}")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        print(f"GPU (CUDA): Available - {torch.cuda.get_device_name(0)}")
    else:
        print("GPU (CUDA): Not Available")
        
    if torch.backends.mps.is_available():
        devices.append("mps")
        print("GPU (MPS): Available (Apple Silicon / Metal)")
    else:
        print("GPU (MPS): Not Available")
        
    print("=" * 50)
    return devices

def run_benchmark(device: str):
    print(f"\n🚀 Preparing Benchmark for Device: {device.upper()}...")
    
    # 1. Load small data chunk for benchmark (10k rows)
    df = load_crypto_data(symbol="BTCUSDT", start_date="2024-01-01", end_date="2024-06-01", interval="4h", use_cache=True)
    if len(df) > 10000:
        df = df.iloc[-10000:]
        
    # 2. Features
    hmm_path = get_or_train_hmm()
    fg = FeatureGenerator(hmm_path=hmm_path)
    processed_df = fg.transform(df)
    processed_df['asset_idx'] = 0
    
    # 3. Env
    env = DummyVecEnv([lambda: TradingEnvV6(df=processed_df, total_assets=3, t_max=1440)])
    env = VecFrameStack(env, n_stack=N_STACK)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # 4. Model Setup
    hmm_cols = [c for c in processed_df.columns if 'hmm_regime' in c]
    num_hmm_states = len(hmm_cols)
    
    policy_kwargs = dict(
        features_extractor_class=GatedTransformerExtractor,
        features_extractor_kwargs=dict(features_dim=128, num_hmm_states=num_hmm_states, n_stack=N_STACK),
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )
    
    print(f"⌛ Starting {TIMESTEPS} steps of PPO (Transformer) on {device.upper()}...")
    
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=1024,
        batch_size=64,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=0
    )
    
    start_time = time.time()
    model.learn(total_timesteps=TIMESTEPS)
    end_time = time.time()
    
    duration = end_time - start_time
    fps = TIMESTEPS / duration
    
    print("-" * 50)
    print(f"✅ Device: {device.upper()}")
    print(f"⏱️  Time taken: {duration:.2f} seconds")
    print(f"⚡ Speed: {fps:.2f} steps/second (FPS)")
    print("-" * 50)
    
    return fps

if __name__ == "__main__":
    devices = get_system_info()
    
    results = {}
    for d in devices:
        try:
            fps = run_benchmark(d)
            results[d] = fps
        except Exception as e:
            print(f"❌ Failed to run on {d.upper()}: {e}")
            
    print("\n🏆 BENCHMARK RESULTS (Transformer PPO):")
    for d, fps in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {d.upper():<5} -> {fps:>8.2f} FPS")
