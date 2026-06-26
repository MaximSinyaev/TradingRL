import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TradingMetricsCallback(BaseCallback):
    """
    Custom callback for logging real PnL and other environment metrics to TensorBoard and Weights & Biases.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.pnls = []
        self.rewards = []
        
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "pnl" in info:
                self.pnls.append(info["pnl"])
            if "real_reward" in info:
                self.rewards.append(info["real_reward"])
        return True
        
    def _on_rollout_end(self) -> None:
        import wandb
        if len(self.pnls) > 0:
            mean_pnl = np.mean(self.pnls)
            mean_reward = np.mean(self.rewards) if len(self.rewards) > 0 else 0.0
            
            self.logger.record("env/mean_pnl", mean_pnl)
            self.logger.record("env/mean_real_reward", mean_reward)
                
            self.pnls = []
            self.rewards = []
