import numpy as np
import pandas as pd
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback
from custom_envs.trading_env_v6 import TradingEnvV6

class TradingMetricsCallback(BaseCallback):
    """
    Custom callback for logging real PnL and other environment metrics to TensorBoard and Weights & Biases.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.pnls = []
        self.rewards = []
        self.target_weights = []
        self.executed_trades = []
        self.margin_calls = []
        self.drawdowns = []
        
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "pnl" in info:
                self.pnls.append(info["pnl"])
            if "real_reward" in info:
                self.rewards.append(info["real_reward"])
            if "target_weight" in info:
                self.target_weights.append(info["target_weight"])
            if "executed" in info:
                self.executed_trades.append(float(info["executed"]))
            if "margin_call" in info:
                self.margin_calls.append(float(info["margin_call"]))
            if "drawdown" in info:
                self.drawdowns.append(info["drawdown"])
        return True
        
    def _on_rollout_end(self) -> None:
        if len(self.pnls) > 0:
            mean_pnl = np.mean(self.pnls)
            mean_reward = np.mean(self.rewards) if len(self.rewards) > 0 else 0.0
            
            self.logger.record("env/mean_pnl", mean_pnl)
            self.logger.record("env/mean_real_reward", mean_reward)
            
            if len(self.target_weights) > 0:
                self.logger.record("action/mean_weight", np.mean(self.target_weights))
                self.logger.record("action/variance", np.var(self.target_weights))
                self.logger.record("action/std_dev", np.std(self.target_weights))
                
            if len(self.executed_trades) > 0:
                self.logger.record("env/trade_frequency", np.mean(self.executed_trades))
                
            if len(self.margin_calls) > 0:
                self.logger.record("env/margin_call_rate", np.mean(self.margin_calls))
                
            if len(self.drawdowns) > 0:
                self.logger.record("env/mean_drawdown", np.mean(self.drawdowns))
                # Поскольку просадка отрицательная (например, -10%), берем минимум
                self.logger.record("env/worst_drawdown", np.min(self.drawdowns))
                
            self.pnls = []
            self.rewards = []
            self.target_weights = []
            self.executed_trades = []
            self.margin_calls = []
            self.drawdowns = []


class OOSEvalCallback(BaseCallback):
    """Out-Of-Sample Evaluation Callback with Fixed Regime-Based Slices (Multi-Asset).

    This callback evaluates the model on 6 predefined validation slices covering
    3 market regimes (Bull, Bear, Flat) with 2 examples each, across multiple assets.

    Validation is deterministic: domain_randomization=False, t_max=None.

    Metrics are 2-dimensional:
    1. By regime (all assets together): val/bull_pnl_mean, val/bull_sortino, etc.
    2. By asset (all regimes together): val/BTC_pnl_mean, val/BTC_sortino, etc.

    Args:
        val_dfs: Dict of asset_name to OOS DataFrame with datetime index or timestamp column
            Example: {"BTCUSDT": btc_df, "ETHUSDT": eth_df, ...}
        val_slices: Dict of slice names to (start_date, end_date) tuples
            Example: {"bull_1": ("2024-01-15", "2024-02-28"), ...}
        eval_freq: Evaluate every N training steps
        experiment_name: Experiment name for logging
        best_model_save_path: Path to save best model (best by sortino_overall)
        verbose: Verbosity level (0: silent, 1: progress bars)

    Environment params (for creating validation environments):
        initial_deposit, commission, leverage, base_slippage, volatility_factor,
        downside_penalty, max_drawdown_pct, weight_tolerance
    """

    def __init__(
        self,
        eval_envs_dict: dict,
        eval_freq: int = 50000,
        experiment_name: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)

        self.eval_freq = eval_freq
        self.experiment_name = experiment_name
        self.best_model_save_path = best_model_save_path

        # Best model tracking
        self.best_sortino = -np.inf
        self.best_model = None

        self.val_envs = eval_envs_dict
        
        # Deduce asset_names and val_slices from eval_envs_dict keys
        self.asset_names = list(set([k[0] for k in self.val_envs.keys()]))
        self.val_slices = {k[1]: None for k in self.val_envs.keys()}

    def _init_callback(self) -> None:
        pass

    def _evaluate_on_slice(self, env: TradingEnvV6) -> dict:
        """Evaluate model on a single validation slice.

        Args:
            env: Validation environment for this slice

        Returns:
            Dict with metrics: pnl, sortino, num_steps
        """
        obs = env.reset()
        done = False
        step_count = 0

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = dones[0]
            step_count += 1

        # Get final metrics from info (infos is a list of dicts for VecEnv)
        info = infos[0]
        pnl = info.get("pnl", 0.0)
        sortino = info.get("episode_sortino", 0.0)

        return {
            "pnl": pnl,
            "sortino": sortino,
            "num_steps": step_count
        }

    def _on_step(self) -> bool:
        """Called at each training step.

        Returns:
            bool: If True, training is stopped (not used here).
        """
        # Check if it's time to evaluate
        if self.num_timesteps % self.eval_freq != 0:
            return True

        if self.verbose > 0:
            print(f"\n=== OOSEvalCallback: Evaluation at step {self.num_timesteps} ===")

        # Evaluate on all (asset, slice) combinations
        results = {}  # (asset, slice) -> {pnl, sortino, num_steps}
        for (asset_name, slice_name), env in self.val_envs.items():
            result = self._evaluate_on_slice(env)
            results[(asset_name, slice_name)] = result

            if self.verbose > 0:
                print(f"{asset_name}/{slice_name}: PnL={result['pnl']:.2f}%, Sortino={result['sortino']:.4f}")

        # --- 1. Aggregate by regime (all assets together) ---
        regime_results = {}
        for regime in ["bull", "bear", "flat"]:
            regime_slices = [slice_name for slice_name in self.val_slices.keys()
                             if slice_name.startswith(regime)]
            if regime_slices:
                regime_pnls = []
                regime_sortinos = []
                for asset_name in self.asset_names:
                    for slice_name in regime_slices:
                        if (asset_name, slice_name) in results:
                            regime_pnls.append(results[(asset_name, slice_name)]["pnl"])
                            regime_sortinos.append(results[(asset_name, slice_name)]["sortino"])

                if regime_pnls:
                    regime_results[f"{regime}_pnl_mean"] = np.mean(regime_pnls)
                    regime_results[f"{regime}_sortino"] = np.mean(regime_sortinos)

        # --- 2. Aggregate by asset (all regimes together) ---
        asset_results = {}
        for asset_name in self.asset_names:
            asset_pnls = []
            asset_sortinos = []
            for slice_name in self.val_slices.keys():
                if (asset_name, slice_name) in results:
                    asset_pnls.append(results[(asset_name, slice_name)]["pnl"])
                    asset_sortinos.append(results[(asset_name, slice_name)]["sortino"])

            if asset_pnls:
                # Use short asset name for logging (e.g., "BTC" instead of "BTCUSDT")
                short_name = asset_name.replace("USDT", "").replace("BUSD", "")
                asset_results[f"{short_name}_pnl_mean"] = np.mean(asset_pnls)
                asset_results[f"{short_name}_sortino"] = np.mean(asset_sortinos)

        # --- 3. Calculate overall metrics ---
        all_pnls = [r["pnl"] for r in results.values()]
        all_sortinos = [r["sortino"] for r in results.values()]

        overall_pnl = np.sum(all_pnls)
        overall_sortino = np.mean(all_sortinos)
        ep_returns_std = np.std(all_pnls)

        # --- 4. Log all metrics ---

        # Log per-regime metrics
        for key, value in regime_results.items():
            self.logger.record(f"val/{key}", value)

        # Log per-asset metrics
        for key, value in asset_results.items():
            self.logger.record(f"val/{key}", value)

        # Log overall metrics
        self.logger.record("val/pnl_total", overall_pnl)
        self.logger.record("val/sortino_overall", overall_sortino)
        self.logger.record("val/ep_returns_std", ep_returns_std)

        if self.verbose > 0:
            print(f"\nBy Regime (all assets):")
            for regime in ["bull", "bear", "flat"]:
                if f"{regime}_pnl_mean" in regime_results:
                    print(f"  {regime.capitalize()}: PnL={regime_results[f'{regime}_pnl_mean']:.2f}%, "
                          f"Sortino={regime_results[f'{regime}_sortino']:.4f}")

            print(f"\nBy Asset (all regimes):")
            for asset_name in self.asset_names:
                short_name = asset_name.replace("USDT", "").replace("BUSD", "")
                if f"{short_name}_pnl_mean" in asset_results:
                    print(f"  {short_name}: PnL={asset_results[f'{short_name}_pnl_mean']:.2f}%, "
                          f"Sortino={asset_results[f'{short_name}_sortino']:.4f}")

            print(f"\nOverall: PnL={overall_pnl:.2f}%, Sortino={overall_sortino:.4f}, Std={ep_returns_std:.2f}")

        # Save best model
        if overall_sortino > self.best_sortino:
            self.best_sortino = overall_sortino

            if self.best_model_save_path:
                self.model.save(self.best_model_save_path)

                if self.verbose > 0:
                    print(f"✅ New best model saved! Sortino: {overall_sortino:.4f}")

        return True
