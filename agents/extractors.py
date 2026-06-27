import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GatedMlpExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128, num_hmm_states: int = 3):
        super().__init__(observation_space, features_dim)
        self.num_hmm_states = num_hmm_states
        self.market_features_dim = observation_space.shape[0] - num_hmm_states
        
        self.proj = nn.Sequential(
            nn.Linear(self.market_features_dim, 256),
            nn.ReLU()
        )
        self.film_layer = FiLMLayer(cond_dim=self.num_hmm_states, feature_dim=256, batch_first=True)
        
        self.out_net = nn.Sequential(
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        market_obs = observations[:, :-self.num_hmm_states]
        hmm_probs = observations[:, -self.num_hmm_states:]
        
        x = self.proj(market_obs)
        x = self.film_layer(x, hmm_probs)
        market_embeddings = self.out_net(x)
        return market_embeddings

class GatedCnnExtractor(BaseFeaturesExtractor):
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
            nn.ReLU()
        )
        
        # We apply FiLM to the channels of the CNN output (out_channels=64)
        self.film_layer = FiLMLayer(cond_dim=self.num_hmm_states, feature_dim=64, batch_first=True)
        
        self.flatten = nn.Flatten()
        cnn_out_dim = 64 * n_stack
        
        self.out_net = nn.Sequential(
            nn.Linear(cnn_out_dim, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        obs_reshaped = observations.view(batch_size, self.n_stack, self.orig_dim)
        
        market_obs = obs_reshaped[:, :, :-self.num_hmm_states]
        hmm_probs = obs_reshaped[:, -1, -self.num_hmm_states:]
        
        # CNN expects (Batch, Channels, Time)
        market_obs_cnn = market_obs.permute(0, 2, 1)
        
        x = self.cnn(market_obs_cnn) # (Batch, 64, Time)
        
        # FiLM on channel dimension. Need to reshape for FiLMLayer: x should have feature in last dim for our FiLMLayer!
        # Our FiLMLayer expects last dimension to be feature_dim.
        # x is (Batch, 64, Seq). We transpose to (Batch, Seq, 64), apply FiLM, then transpose back or just flatten.
        x = x.permute(0, 2, 1) # (Batch, Seq, 64)
        x = self.film_layer(x, hmm_probs)
        
        x = self.flatten(x)
        market_embeddings = self.out_net(x)
        
        return market_embeddings

class GatedGruExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128, num_hmm_states: int = 3, n_stack: int = 10):
        super().__init__(observation_space, features_dim)
        self.n_stack = n_stack
        self.orig_dim = observation_space.shape[0] // n_stack
        self.num_hmm_states = num_hmm_states
        self.market_features_dim = self.orig_dim - num_hmm_states
        
        # Project market obs before GRU to apply FiLM
        self.proj = nn.Linear(self.market_features_dim, 64)
        self.film_layer = FiLMLayer(cond_dim=self.num_hmm_states, feature_dim=64, batch_first=True)
        
        self.gru = nn.GRU(input_size=64, hidden_size=features_dim, batch_first=True)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        obs_reshaped = observations.view(batch_size, self.n_stack, self.orig_dim)
        
        market_obs = obs_reshaped[:, :, :-self.num_hmm_states]
        hmm_probs = obs_reshaped[:, -1, -self.num_hmm_states:]
        
        x = self.proj(market_obs) # (Batch, Seq, 64)
        x = self.film_layer(x, hmm_probs) # FiLM modulates the input sequence
        
        _, hidden = self.gru(x)
        market_embeddings = hidden.squeeze(0)
        
        return market_embeddings

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Conditions the input features based on auxiliary conditioning data (e.g. HMM probabilities).
    """
    def __init__(self, cond_dim: int, feature_dim: int, batch_first: bool = False):
        super().__init__()
        self.batch_first = batch_first
        self.film_gen = nn.Linear(cond_dim, feature_dim * 2)
        # Initialize to identity transform (gamma=0, beta=0)
        nn.init.zeros_(self.film_gen.weight)
        nn.init.zeros_(self.film_gen.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x shape: (Seq, Batch, Feature) or (Batch, Seq, Feature) or (Batch, Feature)
        # cond shape: (Batch, Cond_Dim)
        
        film_params = self.film_gen(cond) # (Batch, feature_dim * 2)
        gamma, beta = film_params.chunk(2, dim=-1) # (Batch, feature), (Batch, feature)
        
        if x.dim() == 3:
            if self.batch_first:
                # Broadcast to (Batch, Seq, Feature)
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
            else:
                # Broadcast to (Seq, Batch, Feature)
                gamma = gamma.unsqueeze(0)
                beta = beta.unsqueeze(0)
            
        return x * (1 + gamma) + beta


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
        
        self.film_layer = FiLMLayer(cond_dim=self.num_hmm_states, feature_dim=d_model)
        
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
        
        # Apply FiLM Conditioning on early features!
        x = self.film_layer(x, hmm_probs)
        
        x = self.pos_encoder(x)
        
        # Pass through transformer
        out = self.transformer(x) # (Seq, Batch, d_model)
        
        # Take the output of the last frame (most recent)
        last_out = out[-1, :, :] # (Batch, d_model)
        
        market_embeddings = self.output_proj(last_out)
        
        return market_embeddings

