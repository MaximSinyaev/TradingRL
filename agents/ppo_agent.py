import os
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

class GatedMlpExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor that uses HMM probabilities as a Gating Mechanism (FiLM).
    Does NOT use Frame Stacking (n_stack=1).
    """
    def __init__(self, observation_space, features_dim: int = 128, num_hmm_states: int = 3):
        super().__init__(observation_space, features_dim)
        self.num_hmm_states = num_hmm_states
        self.market_features_dim = observation_space.shape[0] - num_hmm_states
        
        self.market_net = nn.Sequential(
            nn.Linear(self.market_features_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, features_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.gate_net = nn.Sequential(
            nn.Linear(self.num_hmm_states, features_dim),
            nn.Sigmoid()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        market_obs = observations[:, :-self.num_hmm_states]
        hmm_probs = observations[:, -self.num_hmm_states:]
        market_embeddings = self.market_net(market_obs)
        gates = self.gate_net(hmm_probs)
        return market_embeddings * gates


class GatedCnnExtractor(BaseFeaturesExtractor):
    """
    1D CNN Feature Extractor with Frame Stacking and HMM Gating.
    """
    def __init__(self, observation_space, features_dim: int = 128, num_hmm_states: int = 3, n_stack: int = 10):
        super().__init__(observation_space, features_dim)
        self.n_stack = n_stack
        self.orig_dim = observation_space.shape[0] // n_stack
        self.num_hmm_states = num_hmm_states
        self.market_features_dim = self.orig_dim - num_hmm_states
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.market_features_dim, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        cnn_out_dim = 64 * n_stack
        self.market_net = nn.Sequential(
            self.cnn,
            nn.Linear(cnn_out_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.gate_net = nn.Sequential(
            nn.Linear(self.num_hmm_states, features_dim),
            nn.Sigmoid()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        # Reshape: (Batch, n_stack, orig_dim)
        obs_reshaped = observations.view(batch_size, self.n_stack, self.orig_dim)
        
        # (Batch, n_stack, market_dim)
        market_obs = obs_reshaped[:, :, :-self.num_hmm_states]
        # HMM from the latest step
        hmm_probs = obs_reshaped[:, -1, -self.num_hmm_states:]
        
        # CNN expects (Batch, Channels, Time)
        market_obs_cnn = market_obs.permute(0, 2, 1)
        market_embeddings = self.market_net(market_obs_cnn)
        gates = self.gate_net(hmm_probs)
        
        return market_embeddings * gates


class GatedGruExtractor(BaseFeaturesExtractor):
    """
    GRU Feature Extractor with Frame Stacking and HMM Gating.
    """
    def __init__(self, observation_space, features_dim: int = 128, num_hmm_states: int = 3, n_stack: int = 10):
        super().__init__(observation_space, features_dim)
        self.n_stack = n_stack
        self.orig_dim = observation_space.shape[0] // n_stack
        self.num_hmm_states = num_hmm_states
        self.market_features_dim = self.orig_dim - num_hmm_states
        
        self.gru = nn.GRU(input_size=self.market_features_dim, hidden_size=features_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        
        self.gate_net = nn.Sequential(
            nn.Linear(self.num_hmm_states, features_dim),
            nn.Sigmoid()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        obs_reshaped = observations.view(batch_size, self.n_stack, self.orig_dim)
        
        market_obs = obs_reshaped[:, :, :-self.num_hmm_states]
        hmm_probs = obs_reshaped[:, -1, -self.num_hmm_states:]
        
        # GRU returns (output, hidden)
        _, hidden = self.gru(market_obs)
        # hidden shape: (1, Batch, hidden_size)
        market_embeddings = hidden.squeeze(0)
        market_embeddings = self.dropout(market_embeddings)
        
        gates = self.gate_net(hmm_probs)
        return market_embeddings * gates


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
            
            if wandb.run is not None:
                wandb.log({"env/mean_pnl": mean_pnl, "env/mean_real_reward": mean_reward}, commit=False)
                
            self.pnls = []
            self.rewards = []


def create_ppo_agent(
    env, 
    extractor_type: str = "mlp", 
    num_hmm_states: int = 3, 
    n_stack: int = 1,
    tensorboard_log: str = "./tensorboard_logs/"
):
    """
    Creates a PPO agent with the specified environment and extractor type ('mlp', 'cnn', 'gru').
    If extractor_type is 'cnn' or 'gru', env must be wrapped in VecFrameStack with `n_stack`.
    """
    policy_kwargs = {}
    
    if extractor_type == "mlp":
        policy_kwargs = dict(
            features_extractor_class=GatedMlpExtractor,
            features_extractor_kwargs=dict(features_dim=128, num_hmm_states=num_hmm_states),
            net_arch=dict(pi=[64, 64], vf=[64, 64])
        )
    elif extractor_type == "cnn":
        policy_kwargs = dict(
            features_extractor_class=GatedCnnExtractor,
            features_extractor_kwargs=dict(features_dim=128, num_hmm_states=num_hmm_states, n_stack=n_stack),
            net_arch=dict(pi=[64, 64], vf=[64, 64])
        )
    elif extractor_type == "gru":
        policy_kwargs = dict(
            features_extractor_class=GatedGruExtractor,
            features_extractor_kwargs=dict(features_dim=128, num_hmm_states=num_hmm_states, n_stack=n_stack),
            net_arch=dict(pi=[64, 64], vf=[64, 64])
        )
    else:
        raise ValueError("extractor_type must be 'mlp', 'cnn', or 'gru'")
        
    if extractor_type == "mlp":
        device = "cpu"
    else:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        
    print(f"🚀 Initializing PPO ({extractor_type.upper()}) on device: {device}")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        device=device,
        verbose=0
    )
    return model
