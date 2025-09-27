"""
Unified Pipeline for Machine Learning Experiments
Dependencies:
    !pip install numpy scipy scikit-learn pandas joblib tqdm transformers torch tensorflow
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from scipy.sparse import save_npz, load_npz
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping
from transformers import AutoTokenizer, AutoModel
import torch

class ExperimentRunner:
    def __init__(self, config: dict, workdir: str = ".", save_models: bool = True, verbose: bool = True):
        self.config = config
        self.workdir = workdir
        self.save_models = save_models
        self.verbose = verbose
        os.makedirs(os.path.join(workdir, "features"), exist_ok=True)
        os.makedirs(os.path.join(workdir, "artifacts"), exist_ok=True)
        os.makedirs(os.path.join(workdir, "results"), exist_ok=True)

    def run_all_experiments(self, X_train, y_train, X_test, y_test) -> pd.DataFrame:
        results = []
        pipelines = self.config.get("sklearn_pipelines", []) + self.config.get("dl_pipelines", [])
        for idx, pipeline in enumerate(pipelines):
            pipeline_id = f"bf{idx}_{pipeline['feature_extractor']['name']}_{pipeline['model']['name']}"
            try:
                result = self.run_single_pipeline(pipeline, X_train, y_train, X_test, y_test, pipeline_id)
                results.append(result)
            except Exception as e:
                print(f"Pipeline {pipeline_id} failed: {e}")
        final_results_df = pd.DataFrame(results).sort_values(by="test_f1_macro", ascending=False)
        final_results_df.to_csv(os.path.join(self.workdir, "results", "final_results.csv"), index=False)
        return final_results_df

    def run_single_pipeline(self, pipeline_config, X_train, y_train, X_test, y_test, pipeline_id):
        start_time = time.time()
        feature_cfg = pipeline_config["feature_extractor"]
        model_cfg = pipeline_config["model"]

        # Load or build features
        X_train_feat, X_test_feat = self.load_or_build_feature(feature_cfg, X_train, X_test, y_train)

        # Train model
        if model_cfg["name"] in ["multinomial_nb", "logistic_regression", "linear_svc"]:
            model, train_time = self.train_sklearn_model(model_cfg, X_train_feat, y_train)
        elif model_cfg["name"] in ["rnn", "lstm"]:
            model, train_time = self.train_dl_model(model_cfg, X_train_feat, y_train)
        else:
            raise ValueError(f"Unsupported model: {model_cfg['name']}")

        # Evaluate model
        inference_start = time.time()
        y_pred = model.predict(X_test_feat)
        inference_time = time.time() - inference_start

        metrics = {
            "test_f1_macro": f1_score(y_test, y_pred, average="macro"),
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision_macro": precision_score(y_test, y_pred, average="macro"),
            "test_recall_macro": recall_score(y_test, y_pred, average="macro"),
        }

        # Save model artifact
        model_path = None
        if self.save_models:
            model_path = os.path.join(self.workdir, "artifacts", f"{pipeline_id}.joblib")
            if model_cfg["name"] in ["multinomial_nb", "logistic_regression", "linear_svc"]:
                joblib.dump(model, model_path)
            elif model_cfg["name"] in ["rnn", "lstm"]:
                model.save(model_path)

        return {
            "pipeline_id": pipeline_id,
            "feature_extractor_name": feature_cfg["name"],
            "model_name": model_cfg["name"],
            "train_time_s": train_time,
            "inference_time_s": inference_time,
            "model_artifact_path": model_path,
            **metrics,
        }

    def load_or_build_feature(self, feature_cfg, X_train, X_test, y_train=None):
        precomputed = feature_cfg.get("precomputed", {})
        if "file" in precomputed and os.path.exists(precomputed["file"]):
            data = np.load(precomputed["file"])
            return data["X_train"], data["X_test"]
        elif "train_file" in precomputed and os.path.exists(precomputed["train_file"]):
            return load_npz(precomputed["train_file"]), load_npz(precomputed["test_file"])

        if feature_cfg["name"] in ["bow", "tfidf"]:
            return self.compute_bow_or_tfidf(feature_cfg["name"], feature_cfg["params"], X_train, X_test)
        elif feature_cfg["name"] == "glove_mean":
            return self.compute_glove_mean(feature_cfg.get("params", {}), X_train, X_test)
        elif feature_cfg["name"] == "bert_cls":
            return self.compute_bert_cls(feature_cfg["params"]["model_name"], X_train, X_test)
        else:
            raise ValueError(f"Unsupported feature extractor: {feature_cfg['name']}")

    def compute_bow_or_tfidf(self, kind, params, X_train_texts, X_test_texts):
        vectorizer = CountVectorizer(**params) if kind == "bow" else TfidfVectorizer(**params)
        X_train_mat = vectorizer.fit_transform(X_train_texts)
        X_test_mat = vectorizer.transform(X_test_texts)
        save_npz(os.path.join(self.workdir, "features", f"{kind}_train.npz"), X_train_mat)
        save_npz(os.path.join(self.workdir, "features", f"{kind}_test.npz"), X_test_mat)
        return X_train_mat, X_test_mat

    def compute_glove_mean(self, params, X_train_texts, X_test_texts):
        glove_path = params.get("glove_txt_path", "glove.6B.100d.txt")
        embedding_dim = params.get("embedding_dim", 100)
        if not os.path.exists(glove_path):
            raise FileNotFoundError(f"GloVe file not found at {glove_path}")
        embeddings = {}
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                embeddings[word] = vector
        def embed_text(texts):
            return np.array([
                np.mean([embeddings[word] for word in text.split() if word in embeddings] or [np.zeros(embedding_dim)], axis=0)
                for text in texts
            ])
        X_train_vec = embed_text(X_train_texts)
        X_test_vec = embed_text(X_test_texts)
        np.savez_compressed(os.path.join(self.workdir, "features", "glove.npz"), X_train=X_train_vec, X_test=X_test_vec)
        return X_train_vec, X_test_vec

    def compute_bert_cls(self, model_name, X_train_texts, X_test_texts, batch_size=32, device="cpu"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        def embed_texts(texts):
            embeddings = []
            for i in tqdm(range(0, len(texts), batch_size), desc="BERT embedding"):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
            return np.vstack(embeddings)
        X_train_vec = embed_texts(X_train_texts)
        X_test_vec = embed_texts(X_test_texts)
        np.savez_compressed(os.path.join(self.workdir, "features", "bert_static.npz"), X_train=X_train_vec, X_test=X_test_vec)
        return X_train_vec, X_test_vec

    def train_sklearn_model(self, model_cfg, X_train, y_train):
        model_class = {
            "multinomial_nb": "sklearn.naive_bayes.MultinomialNB",
            "logistic_regression": "sklearn.linear_model.LogisticRegression",
            "linear_svc": "sklearn.svm.LinearSVC",
        }[model_cfg["name"]]
        model = eval(model_class)()
        hpo_space = model_cfg.get("hpo_space", {})
        search = RandomizedSearchCV(model, hpo_space, n_iter=self.config["settings"]["hpo_settings"]["n_iter"],
                                    cv=self.config["settings"]["hpo_settings"]["cv_folds"], scoring="f1_macro", n_jobs=-1)
        start_time = time.time()
        search.fit(X_train, y_train)
        train_time = time.time() - start_time
        return search.best_estimator_, train_time

    def train_dl_model(self, model_cfg, X_train, y_train):
        num_classes = len(np.unique(y_train))
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(128, return_sequences=False),
            Dense(num_classes, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        start_time = time.time()
        model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32, callbacks=[early_stopping], verbose=1)
        train_time = time.time() - start_time
        return model, train_time

if __name__ == "__main__":
    print("Unified Pipeline Module Loaded.")
    print("Example usage:")
    print("""
    from unified_pipeline import ExperimentRunner, EXPERIMENT_CONFIG
    runner = ExperimentRunner(config=EXPERIMENT_CONFIG)
    final_results_df = runner.run_all_experiments(X_train, y_train, X_test, y_test)
    """)
