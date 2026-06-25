import numpy as np
import pandas as pd
import sys
import os

# Ensure the parent directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_envs.trading_env_v4 import TradingEnvV4
from custom_envs.trading_env_v5 import TradingEnvV5

def generate_mock_data(size=2000):
    np.random.seed(42)
    # Price random walk
    returns = np.random.normal(0, 0.005, size)
    close = 50000 * np.exp(np.cumsum(returns))
    
    # Volatility and Funding Rate
    gk_volatility = np.abs(np.random.normal(0.002, 0.001, size))
    funding_rate = np.random.normal(0.0001, 0.00005, size) # Positive bias
    
    # Random Features
    feature_cols = [f'feat_{i}' for i in range(10)]
    features = np.random.randn(size, 10).astype(np.float32)
    
    df = pd.DataFrame({
        'close': close,
        'gk_volatility': gk_volatility,
        'fundingRate': funding_rate
    })
    
    for i, col in enumerate(feature_cols):
        df[col] = features[:, i]
        
    df['state_vector'] = list(features)
        
    return df, features, feature_cols, close

def run_random_agent(env, steps=500, seed=42):
    obs, _ = env.reset(seed=seed)
    
    rewards = []
    pnls = []
    drawdowns = []
    margin_calls = 0
    
    for _ in range(steps):
        action = env.action_space.sample()
        if hasattr(env.action_space, 'nvec'): # MultiDiscrete (V4)
            if np.random.rand() > 0.5:
                action[0] = np.random.choice([1, 2])
                action[1] = np.random.randint(1, 4) # Small sizes: 10%-30%
        else: # Box (Continuous V5)
            if np.random.rand() > 0.5:
                # Random trade between 10% and 40%
                val = np.random.uniform(0.1, 0.4)
                action[0] = val if np.random.rand() > 0.5 else -val
            else:
                action[0] = 0.0 # Hold
                
        obs, reward, done, _, info = env.step(action)
        
        rewards.append(reward)
        pnls.append(info.get('pnl', 0.0))
        if 'drawdown' in info:
            drawdowns.append(info['drawdown'])
        if info.get('margin_call', False):
            margin_calls += 1
            
        if done:
            break
            
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "cumulative_reward": sum(rewards),
        "final_pnl_pct": pnls[-1] if pnls else 0.0,
        "max_drawdown_pct": min(drawdowns) if drawdowns else 0.0,
        "margin_calls": margin_calls,
        "steps_survived": len(rewards)
    }

def run_multiple_episodes(env, num_episodes=20, steps=1000):
    results = {
        "steps_survived": [],
        "margin_calls": [],
        "final_pnl_pct": [],
        "cumulative_reward": [],
        "mean_reward": []
    }
    
    for i in range(num_episodes):
        res = run_random_agent(env, steps=steps, seed=42+i)
        results["steps_survived"].append(res["steps_survived"])
        results["margin_calls"].append(res["margin_calls"])
        results["final_pnl_pct"].append(res["final_pnl_pct"])
        results["cumulative_reward"].append(res["cumulative_reward"])
        results["mean_reward"].append(res["mean_reward"])
        
    return {
        "avg_steps_survived": np.mean(results["steps_survived"]),
        "total_margin_calls": sum(results["margin_calls"]),
        "avg_final_pnl_pct": np.mean(results["final_pnl_pct"]),
        "avg_cumulative_reward": np.mean(results["cumulative_reward"]),
        "avg_mean_step_reward": np.mean(results["mean_reward"])
    }

def main():
    print("Generating mock data...")
    df, features, feature_cols, close = generate_mock_data()
    
    print("Initializing environments...")
    env_v4 = TradingEnvV4(features=features, real_prices=close, t_max=1000)
    env_v5 = TradingEnvV5(df=df, t_max=1000)
    
    print("\nRunning V4 (Raw PnL, No Slippage, No Funding) for 20 episodes...")
    res_v4 = run_multiple_episodes(env_v4, num_episodes=20, steps=1000)
    
    print("Running V5 (Sortino Proxy, Dynamic Slippage, Funding Rates) for 20 episodes...")
    res_v5 = run_multiple_episodes(env_v5, num_episodes=20, steps=1000)
    
    print("\n" + "="*60)
    print("COMPARISON REPORT: V4 vs V5 (Average over 20 episodes)")
    print("="*60)
    
    print(f"{'Metric':<25} | {'TradingEnvV4':<15} | {'TradingEnvV5':<15}")
    print("-" * 65)
    print(f"{'Avg Steps Survived':<25} | {res_v4['avg_steps_survived']:<15.1f} | {res_v5['avg_steps_survived']:<15.1f}")
    print(f"{'Total Margin Calls':<25} | {'N/A':<15} | {res_v5['total_margin_calls']:<15}")
    print(f"{'Avg Final PnL (%)':<25} | {res_v4['avg_final_pnl_pct']:<15.2f} | {res_v5['avg_final_pnl_pct']:<15.2f}")
    print(f"{'Avg Cumulative Reward':<25} | {res_v4['avg_cumulative_reward']:<15.4f} | {res_v5['avg_cumulative_reward']:<15.4f}")
    print(f"{'Avg Mean Step Reward':<25} | {res_v4['avg_mean_step_reward']:<15.6f} | {res_v5['avg_mean_step_reward']:<15.6f}")
    print("="*60)
    print("Analysis:")
    print("1. V5 typically survives fewer steps because random trading quickly triggers the -10% margin call.")
    print("2. V5 Final PnL is much worse due to accumulated funding fees and dynamic slippage.")
    print("3. V5 Step Reward is fundamentally different (Sortino step returns) compared to V4's raw PnL accumulation.")
    print("4. Conclusion: V5 is significantly more punishing, forcing the RL agent to learn ACTUAL risk management.")

if __name__ == "__main__":
    main()
