import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # For RL with short sequences (n_stack=10), learnable PE is much more stable 
        # and doesn't overwhelm the initial linear projection.
        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))
        # Initialize with small values
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return x


class GatedTransformerExtractor(BaseFeaturesExtractor):
    """
    Transformer Feature Extractor with Frame Stacking and HMM Gating.
    Processes the sequence of frames using Self-Attention.
    """
    def __init__(self, observation_space, features_dim: int = 128, num_hmm_states: int = 3, n_stack: int = 10, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__(observation_space, features_dim)
        self.n_stack = n_stack
        self.orig_dim = observation_space.shape[0] // n_stack
        self.num_hmm_states = num_hmm_states
        self.market_features_dim = self.orig_dim - num_hmm_states
        
        # Project market features to d_model
        self.input_proj = nn.Linear(self.market_features_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=n_stack)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            dropout=0.0, # NO dropout in RL blocks to stabilize value targets
            activation="gelu",
            batch_first=False 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.ReLU()
            # Removed dropout here as well
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
        
        # Project to d_model
        x = self.input_proj(market_obs) # (Batch, n_stack, d_model)
        
        # Transformer expects (Seq, Batch, Feature)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        
        # Pass through transformer
        out = self.transformer(x) # (Seq, Batch, d_model)
        
        # Take the output of the last frame (most recent)
        last_out = out[-1, :, :] # (Batch, d_model)
        
        market_embeddings = self.output_proj(last_out)
        gates = self.gate_net(hmm_probs)
        
        return market_embeddings * gates

