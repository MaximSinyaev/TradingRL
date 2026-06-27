import os
from pathlib import Path

def get_or_train_hmm(model_path: str = None) -> str:
    """
    Checks if the HMM model exists at the specified path. 
    If not, it automatically runs the training script and saves it there.
    Returns the absolute path to the model.
    """
    if model_path is None:
        # Default to models/hmm_model.pkl relative to the project root
        model_path = str(Path(__file__).parent.parent.parent / "models" / "hmm_model.pkl")
    
    if not os.path.exists(model_path):
        print(f"⚠️ HMM model not found at {model_path}. Starting automatic training...")
        
        # Import train_hmm dynamically to avoid circular imports
        from scripts.train_hmm import train_hmm
        
        # Ensure the directory exists
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Train and save to the specific path
        train_hmm(save_path=model_path)
        
    return model_path
