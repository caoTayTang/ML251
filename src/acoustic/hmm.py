import os
# from hmmlearn import hmm
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from src.constants import MODELS_CKPT

import src.acoustic.hmm_model as hmm

class HMMTrainer:
    """
    Hidden Markov Model trainer for isolated word/digit recognition.
    
    Workflow:
    1. Prepare dataset (MFCC features from FSDD).
    2. Train one GaussianHMM per class (digit).
    3. Save trained models.
    4. Evaluate on test set.
    """
    def __init__(self, n_components=5, n_iter=100, model_path=MODELS_CKPT):
        self.n_components = n_components
        self.n_iter = n_iter
        self.model_path = model_path
        self.models = {}

    def train(self, X_train, y_train, n_classes):
        """
        Train one HMM per class.
        - X_train: list of feature sequences (each shape: [frames, n_mfcc])
        - y_train: list of labels (integers)
        """
        print("[INFO] Training new models...")
        for label in tqdm(range(n_classes), desc="Training HMMs"):
            # collect sequences of this class
            X_label = [X_train[i] for i in range(len(y_train)) if y_train[i] == label]

            # concatenate frames and track lengths
            lengths = [len(x) for x in X_label]
            X_concat = np.concatenate(X_label)

            # train Gaussian HMM
            model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type="diag",
                n_iter=self.n_iter
            )
            model.fit(X_concat, lengths)
            self.models[label] = model

        joblib.dump(self.models, self.model_path)
        print(f"[INFO] Models saved to {self.model_path}")

    def predict(self, X_test, y_test):
        """
        Evaluate HMM models on test set.
        Returns accuracy score.
        """
        y_pred = []
        for feat in X_test:
            scores = {label: model.score(feat) for label, model in self.models.items()}
            pred = max(scores, key=scores.get)
            y_pred.append(pred)
        return y_pred
        # acc = accuracy_score(y_test, y_pred)
        # print(f"[RESULT] Accuracy: {acc*100:.2f}%")
        # return acc
