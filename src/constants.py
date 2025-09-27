from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "audio" / "recordings"
FEATURES_DIR = PROJECT_ROOT / "features" / "mfcc"
MODELS_CKPT = PROJECT_ROOT / "models" / "btl3_acoustic" / "hmm_models.pkl"
