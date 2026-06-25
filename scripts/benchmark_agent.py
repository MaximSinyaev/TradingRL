#!/usr/bin/env python3
"""
Universal benchmark script for trading agents and environments.

Features:
- Benchmark any environment version (V1, V2, V3, V4)
- Compare multiple environments
- Test any agent (Random, DQN, etc.)
- Alignment verification (reward vs PnL)
"""

import sys
sys.path.insert(0, '.')

import argparse
import numpy as np
import pandas as pd
from typing import Any, Optional, List
from pathlib import Path

from core.data.binance_loader import BinanceKlinesDownloader
from core.features.feature_generator import FeatureGenerator
from agents.random_agent import RandomAgent

# Environment registry
ENV_REGISTRY = {
    "v1": ("custom_envs.trading_env_v1", "TradingEnvV1"),
    "v2": ("custom_envs.trading_env_v2", "TradingEnvV2"),
    "v3": ("custom_envs.trading_env_v3", "TradingEnvV3"),
    "v4": ("custom_envs.trading_env_v4", "TradingEnvV4"),
}


def import_env(env_name: str):
    """Dynamic import of environment class."""
    module_name, class_name = ENV_REGISTRY[env_name]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def create_env(env_name: str, features, real_prices, **kwargs):
    """Create environment with version-specific kwargs."""
    env_class = import_env(env_name)

    # Version-specific defaults
    env_kwargs = {
        "features": features,
        "real_prices": real_prices,
        "initial_deposit": kwargs.get("deposit", 10000),
        "commission": kwargs.get("commission", 0.0005),
        "t_max": kwargs.get("t_max", 500),
    }

    # Add env-specific kwargs
    if env_name == "v1":
        env_kwargs["reward_on_trades_only"] = True
    elif env_name == "v3":
        env_kwargs["inactivity_penalty"] = kwargs.get("inactivity_penalty", -0.05)
        env_kwargs["inactivity_threshold"] = kwargs.get("inactivity_threshold", 240)
        env_kwargs["return_pt"] = False
    # V4 has no extra kwargs

    return env_class(**env_kwargs)


def evaluate_agent(
    agent: Any,
    env,
    episodes: int = 50,
    seed: int = 42,
    verbose: bool = False
) -> dict:
    """Evaluate agent on environment.

    Returns:
        Dict with: mean_pnl, std_pnl, mean_reward, trades_counts, etc.
    """
    pnls = []
    rewards = []
    trades_counts = []

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        episode_reward = 0

        while not done:
            # Get valid actions
            possible = env.get_possible_actions()

            # Select action
            action = agent.act(state, possible)
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        pnls.append(env.pnl)
        rewards.append(episode_reward)
        trades_counts.append(len(env.trades))

        if verbose and (ep + 1) % 10 == 0:
            print(f"   Episode {ep+1}/{episodes}: PnL={env.pnl:.2f}%, Reward={episode_reward:.4f}")

    return {
        "mean_pnl": np.mean(pnls),
        "std_pnl": np.std(pnls),
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_trades": np.mean(trades_counts),
        "std_trades": np.std(trades_counts),
        "win_rate": sum(1 for p in pnls if p > 0) / len(pnls) * 100,
        "pnls": pnls,
        "rewards": rewards,
    }


def print_comparison(results: List[dict], env_names: List[str]):
    """Print comparison table."""
    print("\n" + "="*80)
    print("📊 BENCHMARK RESULTS")
    print("="*80)

    rows = []
    for name, res in zip(env_names, results):
        aligned = (res["mean_pnl"] < 0) == (res["mean_reward"] < 0)

        rows.append({
            "Env": name.upper(),
            "PnL (%)": f"{res['mean_pnl']:.2f}",
            "Reward": f"{res['mean_reward']:.4f}",
            "Trades": f"{res['mean_trades']:.0f}",
            "WinRate (%)": f"{res['win_rate']:.0f}",
            "Aligned": "✅" if aligned else "❌"
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print("\n" + "="*80)
    print("🔍 ALIGNMENT CHECK")
    print("="*80)

    for name, res in zip(env_names, results):
        aligned = (res["mean_pnl"] < 0) == (res["mean_reward"] < 0)
        status = "✅ ALIGNED" if aligned else "❌ MISALIGNED"

        print(f"\n{name.upper()}: {status}")
        print(f"  PnL: {res['mean_pnl']:.2f}% → {'LOSING' if res['mean_pnl'] < 0 else 'GAINING'}")
        print(f"  Reward: {res['mean_reward']:.4f} → {'POSITIVE' if res['mean_reward'] > 0 else 'NEGATIVE'}")

        if not aligned:
            print(f"  ⚠️  Agent gets {'positive' if res['mean_reward'] > 0 else 'negative'} reward while losing money!")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Universal trading agent benchmark")
    parser.add_argument(
        "--env",
        nargs="+",
        choices=list(ENV_REGISTRY.keys()),
        default=["v4"],
        help="Environment version(s) to test (can specify multiple for comparison)"
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--start-date", default="2026-06-17", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--t-max", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--deposit", type=float, default=10000, help="Initial deposit")
    parser.add_argument("--commission", type=float, default=0.0005, help="Commission rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print(f"🚀 BENCHMARK: {[e.upper() for e in args.env]}")
    print(f"   Symbol: {args.symbol}")
    print(f"   Episodes: {args.episodes}, T-Max: {args.t_max}")
    print(f"   Seed: {args.seed}")

    # Load data
    print("\n📥 Loading data...")
    loader = BinanceKlinesDownloader(args.symbol)
    data = loader.download(args.start_date, args.end_date)

    print("📊 Generating features...")
    fg = FeatureGenerator()
    features = fg.transform(data)

    feature_states = np.array(features.state_vector.tolist())
    real_prices = features['close'].values

    # Benchmark each env
    all_results = []

    for env_name in args.env:
        print(f"\n{'='*60}")
        print(f"📌 Testing {env_name.upper()}...")
        print(f"{'='*60}")

        env = create_env(
            env_name,
            feature_states[:args.t_max * 2],
            real_prices[:args.t_max * 2],
            deposit=args.deposit,
            commission=args.commission,
            t_max=args.t_max,
        )

        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space.shape}")

        # Create agent (adapt for MultiDiscrete)
        if hasattr(env.action_space, 'n'):  # Discrete
            agent = RandomAgent(env.action_space.n, seed=args.seed)
        else:  # MultiDiscrete
            agent = RandomAgent(None, seed=args.seed)

        # Evaluate
        results = evaluate_agent(
            agent,
            env,
            episodes=args.episodes,
            seed=args.seed,
            verbose=args.verbose
        )

        all_results.append(results)

    # Print comparison
    print_comparison(all_results, args.env)

    print(f"\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
