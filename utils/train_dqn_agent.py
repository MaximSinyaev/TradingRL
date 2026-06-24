"""
Universal DQN training script for any trading environment.

Supports:
- Discrete action spaces (V1, V2, V3)
- MultiDiscrete action spaces (V4)

Usage:
    from agents.dqn_agent import DQNAgent
    from utils.train_dqn_agent import train_dqn_agent

    # Auto-detect from env
    agent = DQNAgent.from_env(env, lr=5e-4)
    rewards, pnls, actions = train_dqn_agent(env, agent, num_episodes=500)
"""

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional, Callable, Any
from IPython.display import clear_output


def adaptive_epsilon_schedule(episode: int, total_episodes: int,
                              epsilon_min: float = 0.05,
                              epsilon_max: float = 1.0) -> float:
    """Adaptive epsilon schedule.

    Decays initially, recovers slightly at end for final exploration.
    """
    decay = np.exp(-5 * episode / total_episodes)
    recovery = (episode / total_episodes) ** 3

    if episode < total_episodes // 2:
        return np.clip(epsilon_min + (epsilon_max - epsilon_min) * (decay + recovery),
                      epsilon_min, epsilon_max)
    else:
        return np.clip(epsilon_min + (epsilon_max - epsilon_min) * (decay + recovery),
                      epsilon_min, epsilon_max - 0.3)


def train_dqn_agent(
    env: Any,
    agent: Any,
    num_episodes: int = 500,
    t_max: Optional[int] = None,
    target_update_freq: Optional[int] = None,
    adaptive_epsilon: bool = False,
    epsilon_min: float = 0.05,
    epsilon_max: float = 1.0,
    render: bool = True,
    save_path: Optional[str] = None,
    eval_every: int = 50,
    eval_episodes: int = 10,
    verbose: bool = True,
) -> tuple:
    """Train DQN agent on trading environment.

    Args:
        env: Trading environment (V1, V2, V3, V4)
        agent: DQN agent (will use from_env if needed)
        num_episodes: Number of training episodes
        t_max: Max steps per episode (None = env limit)
        target_update_freq: Hard update frequency (None = soft update only)
        adaptive_epsilon: Use adaptive epsilon schedule
        epsilon_min: Minimum epsilon
        epsilon_max: Maximum epsilon
        render: Show training plots
        save_path: Path to save model
        eval_every: Evaluate every N episodes
        eval_episodes: Number of episodes for evaluation
        verbose: Print progress

    Returns:
        (rewards, pnls, eval_metrics)
        - rewards: List of total rewards per episode
        - pnls: List of PnL per episode
        - eval_metrics: Dict with evaluation results
    """
    rewards_per_episode = []
    pnls_per_episode = []
    eval_rewards = []
    eval_pnls = []

    for episode in tqdm.trange(num_episodes, desc="Training"):
        # Adaptive epsilon
        if adaptive_epsilon:
            agent.epsilon = adaptive_epsilon_schedule(
                episode, num_episodes,
                epsilon_min=epsilon_min,
                epsilon_max=epsilon_max
            )

        # Reset environment
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        step = 0

        while not done:
            # Get valid actions
            possible_actions = env.get_possible_actions()

            # Select action
            action = agent.act(state, possible_actions=possible_actions)

            # Step
            next_state, reward, done, _, info = env.step(action)

            # Store experience
            agent.store(state, action, reward, next_state, done)

            # Train
            loss = agent.train_step()

            state = next_state
            total_reward += reward
            step += 1

            if t_max and step >= t_max:
                break

        # Track metrics
        rewards_per_episode.append(total_reward)
        pnls_per_episode.append(env.pnl)

        # Decay epsilon at end of episode (if not using adaptive schedule)
        if not adaptive_epsilon:
            agent.decay_epsilon()

        # Hard target update
        if target_update_freq and episode % target_update_freq == 0:
            agent.update_target_network()

        # Evaluation
        if episode % eval_every == 0 and episode > 0:
            eval_reward, eval_pnl = evaluate_agent(agent, env, eval_episodes)
            eval_rewards.append(eval_reward)
            eval_pnls.append(eval_pnl)

            if verbose:
                print(f"\n{'='*60}")
                print(f"Episode {episode}/{num_episodes}")
                print(f"Train - Last 10: Reward={np.mean(rewards_per_episode[-10:]):.4f}, PnL={np.mean(pnls_per_episode[-10:]):.2f}%")
                print(f"Eval  - Mean: Reward={eval_reward:.4f}, PnL={eval_pnl:.2f}%")
                print(f"Epsilon: {agent.epsilon:.3f}")
                print(f"{'='*60}")

        elif verbose and episode % 10 == 0:
            try:
                clear_output(wait=True)
            except:
                pass

            print(f"Episode {episode}/{num_episodes}: "
                  f"Reward={total_reward:.4f}, PnL={env.pnl:.2f}%, "
                  f"Epsilon={agent.epsilon:.3f}")

        # Plotting
        if render and episode % 50 == 0 and episode > 0:
            plot_training_progress(rewards_per_episode, pnls_per_episode,
                                 eval_rewards if eval_rewards else None,
                                 eval_pnls if eval_pnls else None)

    # Save model
    if save_path:
        agent.save_state_dict(save_path)

    eval_metrics = {
        'eval_rewards': eval_rewards,
        'eval_pnls': eval_pnls,
        'eval_every': eval_every,
    }

    return rewards_per_episode, pnls_per_episode, eval_metrics


def evaluate_agent(agent: Any, env: Any, num_episodes: int = 10) -> tuple:
    """Evaluate agent on environment without training.

    Returns:
        (mean_reward, mean_pnl)
    """
    rewards = []
    pnls = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        while not done:
            possible_actions = env.get_possible_actions()
            action = agent.act(state, possible_actions=possible_actions, training=False)
            next_state, reward, done, _, info = env.step(action)

            state = next_state
            total_reward += reward
            step += 1

        rewards.append(total_reward)
        pnls.append(env.pnl)

    return np.mean(rewards), np.mean(pnls)


def plot_training_progress(train_rewards, train_pnls, eval_rewards=None, eval_pnls=None):
    """Plot training progress."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Rewards
    ax = axes[0]
    ax.plot(train_rewards, alpha=0.6, label='Train')
    if eval_rewards:
        eval_x = np.arange(0, len(train_rewards), len(train_rewards) // len(eval_rewards))[:len(eval_rewards)]
        ax.plot(eval_x, eval_rewards, 'o-', label='Eval', markersize=4)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PnL
    ax = axes[1]
    ax.plot(train_pnls, alpha=0.6, label='Train')
    if eval_pnls:
        eval_x = np.arange(0, len(train_pnls), len(train_pnls) // len(eval_pnls))[:len(eval_pnls)]
        ax.plot(eval_x, eval_pnls, 'o-', label='Eval', markersize=4)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("PnL (%)")
    ax.set_title("PnL over Episodes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    from custom_envs.trading_env_v4 import TradingEnvV4
    from agents.dqn_agent import DQNAgent
    from utils.binance_loader import BinanceKlinesDownloader
    from utils.feature_generator import FeatureGenerator

    print("🔄 Loading data...")
    loader = BinanceKlinesDownloader("BTCUSDT")
    data = loader.download("2026-06-17", "2026-06-20")

    print("📊 Generating features...")
    fg = FeatureGenerator()
    features = fg.transform(data)

    feature_states = np.array(features.state_vector.tolist())
    real_prices = features['close'].values

    print("🔧 Creating environment...")
    env = TradingEnvV4(
        features=feature_states[:2000],
        real_prices=real_prices[:2000],
        initial_deposit=10000,
        t_max=500,
    )

    print("🤖 Creating DQN agent...")
    agent = DQNAgent.from_env(env, lr=1e-3)

    print("🏋️ Training...")
    rewards, pnls, _ = train_dqn_agent(
        env, agent,
        num_episodes=100,
        eval_every=20,
        verbose=True
    )

    print(f"\n📊 Final Results:")
    print(f"  Mean reward (last 10): {np.mean(rewards[-10:]):.4f}")
    print(f"  Mean PnL (last 10): {np.mean(pnls[-10:]):.2f}%")
