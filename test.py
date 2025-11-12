import os
import joblib

from src.acoustic.hmm import HMMTrainer
from src.acoustic.dataset import FSDD
from sklearn.model_selection import train_test_split


from pathlib import Path
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_CKPT = "models/btl3_acoustic/scratch_hmm_models.pkl"

if __name__ == "__main__":
    dataset = FSDD()
    X, y = dataset.prepare_data()
    trainer = HMMTrainer()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    trainer = HMMTrainer()
    if os.path.isfile(MODELS_CKPT):
        print(f"[INFO] Loading existing models from {MODELS_CKPT}")
        trainer.models = joblib.load(MODELS_CKPT)
    else:
        trainer.train(X_train, y_train, n_classes=len(dataset.label_to_idx))
    
    y_pred = trainer.evaluate(X_test, y_test)