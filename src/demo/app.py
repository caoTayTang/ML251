import os
import librosa
import numpy as np
import gradio as gr
import joblib

from src.acoustic.dataset import FSDD
from src.acoustic.hmm import HMMTrainer
from src.constants import MODELS_CKPT


# -------------------
# Feature extraction for new audio
# -------------------
def extract_features(file_path, n_mfcc, sr):
    y, _ = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        n_fft=256, hop_length=80
    )
    return mfcc.T


# -------------------
# Prediction
# -------------------
def predict(file_obj):
    print(file_obj)
    # Load trained HMMs
    models = joblib.load(MODELS_CKPT)
    dataset = FSDD()   # just to get label mapping

    mfcc = extract_features(file_obj, n_mfcc=dataset.n_mfcc, sr=dataset.sr)
    scores = {label: model.score(mfcc) for label, model in models.items()}
    pred_idx = max(scores, key=scores.get)
    pred_label = dataset.idx_to_label[pred_idx]

    return f"Predicted Digit: {pred_label}"


# -------------------
# Ensure models exist
# -------------------
def ensure_models():
    if not os.path.exists(MODELS_CKPT):
        print("[INFO] No trained models found. Training...")
        dataset = FSDD()
        X, y = dataset.prepare_data()

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        trainer = HMMTrainer()
        trainer.train(X_train, y_train, n_classes=len(dataset.label_to_idx))
        trainer.evaluate(X_test, y_test)


# -------------------
# Launch Gradio demo
# -------------------
def launch_demo(share=True, debug=True):
    ensure_models()

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
        outputs="text",
        live=False,
        title="🎤 HMM Speech Recognition (Digits)",
        description="Record or upload an audio file of a spoken digit (0-9). The HMM model will predict it."
    )
    demo.launch(share=share, debug=debug)