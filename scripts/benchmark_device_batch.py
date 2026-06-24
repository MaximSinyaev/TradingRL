"""
Benchmark device (CPU vs MPS) and batch size for DoubleDQN training.

Finds optimal configuration for Apple Silicon training.
"""

import time
import numpy as np
import torch
from utils.binance_loader import BinanceKlinesDownloader
from utils.feature_generator import FeatureGenerator
from custom_envs.trading_env_v4 import TradingEnvV4
from agents.double_dqn_agent import DoubleDQNAgent


def benchmark_batch_sizes(device_name, batch_sizes, num_steps=100):
    """Benchmark training speed for different batch sizes."""
    print(f"\n{'='*60}")
    print(f"Device: {device_name}")
    print(f"{'='*60}")

    device = torch.device(device_name)
    results = {}

    for batch_size in batch_sizes:
        # Create agent
        agent = DoubleDQNAgent.from_env(
            env,
            lr=5e-4,
            batch_size=batch_size,
            device=device,
            use_prioritized_replay=False  # Fair comparison
        )

        # Warmup
        for _ in range(10):
            agent.store(states[0], [1, 0], 0.1, states[1], False)
        agent.train_step()

        # Benchmark
        start = time.time()
        for step in range(num_steps):
            agent.store(states[step % len(states)], [1, 0], 0.1, states[(step + 1) % len(states)], False)
            loss = agent.train_step()
        elapsed = time.time() - start

        steps_per_sec = num_steps / elapsed
        results[batch_size] = {
            'elapsed': elapsed,
            'steps_per_sec': steps_per_sec
        }

        print(f"  Batch {batch_size:3d}: {elapsed:5.2f}s ({steps_per_sec:5.1f} steps/sec)")

    return results


def find_optimal_batch(cpu_results, mps_results):
    """Find optimal batch size for MPS vs CPU."""
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")

    # Find fastest for each device
    cpu_best = max(cpu_results.items(), key=lambda x: x[1]['steps_per_sec'])
    mps_best = max(mps_results.items(), key=lambda x: x[1]['steps_per_sec'])

    cpu_batch, cpu_speed = cpu_best
    mps_batch, mps_speed = mps_best

    speedup = mps_speed['steps_per_sec'] / cpu_speed['steps_per_sec']

    print(f"CPU best:   batch={cpu_batch}, {cpu_speed['steps_per_sec']:.1f} steps/sec")
    print(f"MPS best:   batch={mps_batch}, {mps_speed['steps_per_sec']:.1f} steps/sec")
    print(f"Speedup:    {speedup:.2f}x")

    if speedup > 1.1:
        print(f"\n✅ Use MPS with batch_size={mps_batch}")
        return 'mps', mps_batch
    else:
        print(f"\n✅ Use CPU with batch_size={cpu_batch}")
        return 'cpu', cpu_batch


if __name__ == "__main__":
    # Load data (cached)
    print("Loading data...")
    loader = BinanceKlinesDownloader()
    data = loader.download(start_date='2025-03-01', end_date='2025-03-15')

    fg = FeatureGenerator()
    features = fg.transform(data)
    feature_states = np.array(features.state_vector.tolist())

    # Create small env for benchmarking
    N_BENCH = 1000
    env = TradingEnvV4(
        features=feature_states[:N_BENCH],
        real_prices=features['close'].iloc[:N_BENCH].values,
        initial_deposit=10000,
        t_max=N_BENCH - 1,
    )

    # Pre-collect some states for benchmarking
    states = []
    obs, _ = env.reset()
    for _ in range(200):
        action = env.action_space.sample()
        obs, _, done, _, _ = env.step(action)
        states.append(obs)
        if done:
            break
    states = np.array(states)

    # Batch sizes to test (minimal set)
    batch_sizes = [128, 256]

    # Check available devices
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    print(f"\nAvailable devices:")
    print(f"  CUDA: {has_cuda}")
    print(f"  MPS:  {has_mps}")
    print(f"  CPU:  True")

    # Benchmark CPU
    cpu_results = benchmark_batch_sizes('cpu', batch_sizes)

    # Benchmark MPS if available
    if has_mps:
        mps_results = benchmark_batch_sizes('mps', batch_sizes)
    else:
        print("\n⚠️  MPS not available, skipping")
        mps_results = cpu_results  # Fallback

    # Find optimal
    optimal_device, optimal_batch = find_optimal_batch(cpu_results, mps_results)

    print(f"\n{'='*60}")
    print("READY TO TRAIN:")
    print(f"  device = torch.device('{optimal_device}')")
    print(f"  agent = DoubleDQNAgent.from_env(env, batch_size={optimal_batch}, device=device)")
    print(f"{'='*60}")
