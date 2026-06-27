import os
import torch
from stable_baselines3 import PPO

# Import extractors and callbacks from the new submodules
from agents.extractors import GatedMlpExtractor, GatedCnnExtractor, GatedGruExtractor, GatedTransformerExtractor
from agents.callbacks import TradingMetricsCallback

def create_ppo_agent(
    env, 
    extractor_type: str = "mlp", 
    num_hmm_states: int = 3, 
    n_stack: int = 1,
    tensorboard_log: str = "./tensorboard_logs/"
):
    """
    Creates a PPO agent with the specified environment and extractor type ('mlp', 'cnn', 'gru', 'transformer').
    If extractor_type is 'cnn', 'gru' or 'transformer', env must be wrapped in VecFrameStack with `n_stack`.
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
    elif extractor_type == "transformer":
        policy_kwargs = dict(
            features_extractor_class=GatedTransformerExtractor,
            features_extractor_kwargs=dict(features_dim=128, num_hmm_states=num_hmm_states, n_stack=n_stack),
            net_arch=dict(pi=[64, 64], vf=[64, 64])
        )
    else:
        raise ValueError("extractor_type must be 'mlp', 'cnn', 'gru', or 'transformer'")
        
    # Add weight decay for regularization (since we removed dropout)
    policy_kwargs["optimizer_kwargs"] = dict(weight_decay=1e-5)
        
    if extractor_type == "mlp":
        device = "cpu"
    else:
        # We skip 'mps' globally because benchmark showed CPU is 3x faster for our PPO envs
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
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
