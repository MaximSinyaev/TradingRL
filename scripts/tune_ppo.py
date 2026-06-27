import argparse
import optuna
from optuna.pruners import MedianPruner
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from core.data.data_loader import load_crypto_data
from core.features.feature_generator import FeatureGenerator
from core.data.data_splitter import create_purged_train_val_split
from core.config import VAL_SLICES, N_STACK
from custom_envs.trading_env_v6 import TradingEnvV6
from agents.ppo_agent import create_ppo_agent
from agents.callbacks import OOSEvalCallback

# Global variables to hold data
global_train_dfs = None
global_val_dfs_dict = None

def get_data():
    global global_train_dfs, global_val_dfs_dict
    if global_train_dfs is not None:
        return global_train_dfs, global_val_dfs_dict

    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    dfs_dict = {}
    
    # Check or train HMM model automatically
    from core.features.hmm_helper import get_or_train_hmm
    hmm_path = get_or_train_hmm()
    
    fg = FeatureGenerator(hmm_path=hmm_path)
    
    for i, sym in enumerate(symbols):
        print(f"Loading and processing {sym}...")
        df = load_crypto_data(symbol=sym, start_date="2020-01-01", end_date="2026-06-26", interval="4h", use_cache=True)
        processed = fg.transform(df)
        processed['asset_idx'] = i
        dfs_dict[sym] = processed
        
    train_dfs, val_dfs_dict = create_purged_train_val_split(
        dfs_dict=dfs_dict,
        val_slices=VAL_SLICES,
        embargo_candles=42
    )
    
    global_train_dfs = train_dfs
    global_val_dfs_dict = val_dfs_dict
    return train_dfs, val_dfs_dict

class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""
    def __init__(self, eval_env, trial, n_eval_episodes=1, eval_freq=10000, deterministic=True, verbose=0):
        super().__init__(eval_env=eval_env, n_eval_episodes=n_eval_episodes, eval_freq=eval_freq,
                         deterministic=deterministic, verbose=verbose)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            # Assuming mean_reward is computed in super()._on_step()
            mean_reward = self.last_mean_reward
            self.trial.report(mean_reward, self.eval_idx)
            
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

def objective(trial: optuna.Trial):
    train_dfs, val_dfs_dict = get_data()
    
    # 1. Sample Hyperparameters
    extractor_type = trial.suggest_categorical("extractor", ["cnn", "gru", "transformer"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1, log=True)
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.99, 0.995, 0.999])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98, 0.99])
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    
    # Invalid combo check
    if batch_size > n_steps:
        raise optuna.TrialPruned()
        
    import torch
    
    # Use explicitly requested device or fallback to auto-detection
    if args.device != "auto":
        device = args.device
    else:
        # We skip 'mps' because CPU is benchmarked to be 3x-10x faster for our PPO environments
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"\n{'='*50}")
    print(f"🚀 Starting Trial {trial.number}")
    print(f"🖥️  Target Device: {device.upper()}")
    print(f"⚙️ Parameters: {trial.params}")
    print(f"{'='*50}\n")

    import wandb
    from wandb.integration.sb3 import WandbCallback
    import os

    run_name = f"trial_{trial.number}_{extractor_type}_lr{learning_rate:.1e}"
    run = wandb.init(
        project="trading_rl_hpo",
        name=run_name,
        config=trial.params,
        reinit=True,
        sync_tensorboard=True,
    )

    # 2. Setup Environments
    train_env = DummyVecEnv([lambda df=train_dfs: TradingEnvV6(df=df, total_assets=3, t_max=1440)])
    train_env = VecFrameStack(train_env, n_stack=N_STACK)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Create evaluation environments for 4 slices
    eval_envs_list = []
    for slice_name in ["bull_1", "bear_1", "flat_1", "bear_2"]:
        # Capture the dataframe properly in the lambda
        df_slice = val_dfs_dict["BTCUSDT"][slice_name]
        eval_envs_list.append(lambda df=df_slice: TradingEnvV6(df=df, total_assets=3, domain_randomization=False, t_max=None))
        
    eval_env = DummyVecEnv(eval_envs_list)
    eval_env = VecFrameStack(eval_env, n_stack=N_STACK)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False

    # 3. Create Model
    from stable_baselines3 import PPO
    from agents.extractors import GatedTransformerExtractor, GatedGruExtractor, GatedCnnExtractor
    
    if extractor_type == "cnn":
        ext_class = GatedCnnExtractor
    elif extractor_type == "gru":
        ext_class = GatedGruExtractor
    else:
        ext_class = GatedTransformerExtractor
    
    hmm_cols = [c for c in train_dfs[0].columns if 'hmm_regime' in c]
    num_hmm_states = len(hmm_cols)
    
    policy_kwargs = dict(
        features_extractor_class=ext_class,
        features_extractor_kwargs=dict(features_dim=128, num_hmm_states=num_hmm_states, n_stack=N_STACK),
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        optimizer_kwargs=dict(weight_decay=1e-5)
    )
    
    tb_log_dir = os.path.join("tensorboard_logs", "hpo", run_name)
    
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=0.2,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=0,
        tensorboard_log=tb_log_dir
    )
    
    # 4. Train with Eval Pruning (Evaluating every ~100k steps)
    eval_callback = TrialEvalCallback(eval_env, trial, n_eval_episodes=3, eval_freq=100000)
    wandb_callback = WandbCallback(verbose=0)
    
    try:
        model.learn(total_timesteps=1500000, callback=[eval_callback, wandb_callback])
    except (AssertionError, ValueError) as e:
        wandb.finish()
        raise optuna.TrialPruned()
        
    if eval_callback.is_pruned:
        wandb.finish()
        raise optuna.TrialPruned()
        
    reward = eval_callback.last_mean_reward
    wandb.log({"final_val_reward": reward})
    wandb.finish()
    return reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=120)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Force specific device (e.g. cpu)")
    args = parser.parse_args()
    
    print("🚀 Starting PPO Hyperparameter Optimization with Optuna")
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", pruner=pruner, study_name="ppo_transformer_tune")
    
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    
    print("✅ Optimization finished!")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Mean Reward / Proxy Sortino): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
