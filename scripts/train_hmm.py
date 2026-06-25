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
from core.features.feature_generator import FeatureGenerator

def train_hmm():
    print("📥 Loading historical data for HMM training...")
    # Loading enough data to find regimes. 
    # Currently limited by 1000 candles if Assistant 2 hasn't fixed pagination, 
    # but 1000 candles of 4h is ~166 days, which is a good start for a POC.
    df = load_crypto_data(
        symbol="BTCUSDT",
        start_date="2023-01-01",
        interval="4h",
        source="bybit_futures"
    )
    
    if df.empty:
        print("❌ No data loaded. Cannot train HMM.")
        return

    print("📊 Generating base features...")
    fg = FeatureGenerator()
    df_features = fg.transform(df)

    # Select features that define a market regime
    # 1. log_return: defines direction (trend up/down)
    # 2. gk_volatility: defines variance/chop
    # 3. normalized_volume: defines market participation
    obs_cols = ['log_return', 'gk_volatility', 'normalized_volume']
    
    # Check for NaN and clean
    X = df_features[obs_cols].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"🧠 Training HMM on {len(X)} samples using {obs_cols}...")
    
    # Clip extreme outliers (Winsorization: 1st to 99th percentile) to prevent "black swan" clusters
    lower_bounds = X.quantile(0.01)
    upper_bounds = X.quantile(0.99)
    X_clipped = X.clip(lower=lower_bounds, upper=upper_bounds, axis=1)
    
    # Scale features robustly
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_clipped)
    
    # 3 Regimes:
    # - Trend Up (Bull)
    # - Trend Down (Bear)
    # - Chop / Flat (Low vol or high vol chop)
    hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    
    print("⏳ Fitting GaussianHMM (this might take a few seconds)...")
    hmm_model.fit(X_scaled)
    
    # Check convergence
    if hmm_model.monitor_.converged:
        print(f"✅ HMM Converged in {hmm_model.monitor_.iter} iterations.")
    else:
        print("⚠️ HMM did NOT converge. Consider increasing n_iter or checking data scaling.")
        
    # Save the model and scaler
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "hmm_model.pkl"
    joblib.dump({"model": hmm_model, "scaler": scaler}, model_path)
    
    print(f"💾 HMM Model and Scaler saved to {model_path}")
    
    # Optional: print regime characteristics to see what the HMM learned
    hidden_states = hmm_model.predict(X_scaled)
    X['regime'] = hidden_states
    
    print("\n📈 Learned Regime Characteristics (Mean values):")
    for i in range(3):
        mask = X['regime'] == i
        if mask.sum() > 0:
            print(f"Regime {i} ({mask.sum()} samples):")
            print(f"  Log Return: {X.loc[mask, 'log_return'].mean():.6f}")
            print(f"  GK Vol:     {X.loc[mask, 'gk_volatility'].mean():.6f}")
            print(f"  Volume:     {X.loc[mask, 'normalized_volume'].mean():.4f}")

if __name__ == "__main__":
    train_hmm()
