import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler

# Ensure we can import from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data.data_loader import load_crypto_data
from core.data.data_splitter import create_purged_train_val_split
from core.features.feature_generator import FeatureGenerator
from core.config import VAL_SLICES

def train_hmm(save_path: str = None):
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    print(f"📥 Loading historical data for HMM training on {symbols}...")
    
    dfs_dict = {}
    for sym in symbols:
        df = load_crypto_data(
            symbol=sym,
            start_date="2020-01-01",
            end_date="2026-06-26",
            interval="4h",
            source="bybit_futures"
        )
        if not df.empty:
            dfs_dict[sym] = df
            
    if not dfs_dict:
        print("❌ No data loaded. Cannot train HMM.")
        return

    print("✂️ Creating purged train splits to prevent data leakage...")
    train_dfs, _ = create_purged_train_val_split(
        dfs_dict=dfs_dict,
        val_slices=VAL_SLICES,
        embargo_candles=42
    )

    if not train_dfs:
        print("❌ No training data left after purging.")
        return

    print("📊 Generating base features for all training chunks...")
    # Explicitly disable HMM loading during feature generation for training
    fg = FeatureGenerator(hmm_path=None)
    
    all_features = []
    for chunk in train_dfs:
        df_features = fg.transform(chunk)
        all_features.append(df_features)
        
    df_features_combined = pd.concat(all_features, ignore_index=True)

    # Select features that define a market regime
    # 1. log_return: defines direction (trend up/down)
    # 2. gk_volatility: defines variance/chop
    # 3. normalized_volume: defines market participation
    obs_cols = ['log_return', 'gk_volatility', 'normalized_volume']
    
    # Check for NaN and clean
    X = df_features_combined[obs_cols].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"🧠 Training Universal HMM on {len(X)} combined samples using {obs_cols}...")
    
    # Clip extreme outliers (Winsorization: 1st to 99th percentile) to prevent "black swan" clusters
    lower_bounds = X.quantile(0.01)
    upper_bounds = X.quantile(0.99)
    X_clipped = X.clip(lower=lower_bounds, upper=upper_bounds, axis=1)
    
    # Scale features robustly
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_clipped)
    
    # 4 Regimes:
    # - Trend Up (Bull)
    # - Trend Down (Bear)
    # - Chop / Flat (Low vol or high vol chop)
    # - High Volatility (Crash/Climax)
    hmm_model = GaussianHMM(n_components=4, covariance_type="full", n_iter=1000, random_state=42)
    
    print("⏳ Fitting Universal GaussianHMM (this might take a few seconds)...")
    hmm_model.fit(X_scaled)
    
    # Check convergence
    if hmm_model.monitor_.converged:
        print(f"✅ HMM Converged in {hmm_model.monitor_.iter} iterations.")
    else:
        print("⚠️ HMM did NOT converge. Consider increasing n_iter or checking data scaling.")
        
    # Save the model and scaler
    if save_path is None:
        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(exist_ok=True)
        save_path = str(models_dir / "hmm_model.pkl")
    else:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        
    joblib.dump({"model": hmm_model, "scaler": scaler}, save_path)
    
    print(f"💾 Universal HMM Model and Scaler saved to {save_path}")
    
    # Optional: print regime characteristics to see what the HMM learned
    hidden_states = hmm_model.predict(X_scaled)
    X['regime'] = hidden_states
    
    print("\n📈 Learned Regime Characteristics (Mean values):")
    for i in range(4):
        mask = X['regime'] == i
        if mask.sum() > 0:
            print(f"Regime {i} ({mask.sum()} samples):")
            print(f"  Log Return: {X.loc[mask, 'log_return'].mean():.6f}")
            print(f"  GK Vol:     {X.loc[mask, 'gk_volatility'].mean():.6f}")
            print(f"  Volume:     {X.loc[mask, 'normalized_volume'].mean():.4f}")

if __name__ == "__main__":
    train_hmm()
