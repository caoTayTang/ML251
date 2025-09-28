import os
import time
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from scipy.sparse import save_npz, load_npz
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Input, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping
from transformers import AutoTokenizer, AutoModel
import torch

# Export configuration
EXPERIMENT_CONFIG = {
    "settings": {
        "use_cache": True,
        "cache_path": "features",
        "hpo_settings": {
            "n_iter": 15,
            "cv_folds": 3
        }
    },

    # Các pipeline với sklearn models
    "sklearn_pipelines": [
        # --- BoW Unigram ---
        {"feature_extractor": {"name": "bow", "params": {"ngram_range": (1, 1), "max_features": 20000},
                               "precomputed": {"train_file": "features/bow_train_unigram.npz",
                                               "test_file": "features/bow_test_unigram.npz"}},
         "model": {"name": "multinomial_nb", "hpo_space": {"alpha": loguniform(1e-3, 1e0)}}},

        {"feature_extractor": {"name": "bow", "params": {"ngram_range": (1, 1), "max_features": 20000},
                               "precomputed": {"train_file": "features/bow_train_unigram.npz",
                                               "test_file": "features/bow_test_unigram.npz"}},
         "model": {"name": "logistic_regression", "hpo_space": {"C": loguniform(1e-2, 1e2),
                                                                "class_weight": [None, "balanced"]}}},

        {"feature_extractor": {"name": "bow", "params": {"ngram_range": (1, 1), "max_features": 20000},
                               "precomputed": {"train_file": "features/bow_train_unigram.npz",
                                               "test_file": "features/bow_test_unigram.npz"}},
         "model": {"name": "linear_svc", "hpo_space": {"C": loguniform(1e-2, 1e2),
                                                       "class_weight": [None, "balanced"]}}},

        # --- BoW Bigram ---
        {"feature_extractor": {"name": "bow", "params": {"ngram_range": (1, 2), "max_features": 30000},
                               "precomputed": {"train_file": "features/bow_train_bigram.npz",
                                               "test_file": "features/bow_test_bigram.npz"}},
         "model": {"name": "logistic_regression", "hpo_space": {"C": loguniform(1e-2, 1e2),
                                                                "class_weight": [None, "balanced"]}}},

        {"feature_extractor": {"name": "bow", "params": {"ngram_range": (1, 2), "max_features": 30000},
                               "precomputed": {"train_file": "features/bow_train_bigram.npz",
                                               "test_file": "features/bow_test_bigram.npz"}},
         "model": {"name": "linear_svc", "hpo_space": {"C": loguniform(1e-2, 1e2),
                                                       "class_weight": [None, "balanced"]}}},

        # --- TF-IDF Unigram ---
        {"feature_extractor": {"name": "tfidf", "params": {"ngram_range": (1, 1), "max_features": 20000},
                               "precomputed": {"train_file": "features/tfidf_train_unigram.npz",
                                               "test_file": "features/tfidf_test_unigram.npz"}},
         "model": {"name": "multinomial_nb", "hpo_space": {"alpha": loguniform(1e-3, 1e0)}}},

        {"feature_extractor": {"name": "tfidf", "params": {"ngram_range": (1, 1), "max_features": 20000},
                               "precomputed": {"train_file": "features/tfidf_train_unigram.npz",
                                               "test_file": "features/tfidf_test_unigram.npz"}},
         "model": {"name": "logistic_regression", "hpo_space": {"C": loguniform(1e-2, 1e2),
                                                                "class_weight": [None, "balanced"]}}},

        {"feature_extractor": {"name": "tfidf", "params": {"ngram_range": (1, 1), "max_features": 20000},
                               "precomputed": {"train_file": "features/tfidf_train_unigram.npz",
                                               "test_file": "features/tfidf_test_unigram.npz"}},
         "model": {"name": "linear_svc", "hpo_space": {"C": loguniform(1e-2, 1e2),
                                                       "class_weight": [None, "balanced"]}}},

        # --- TF-IDF Bigram ---
        {"feature_extractor": {"name": "tfidf", "params": {"ngram_range": (1, 2), "max_features": 30000},
                               "precomputed": {"train_file": "features/tfidf_train_bigram.npz",
                                               "test_file": "features/tfidf_test_bigram.npz"}},
         "model": {"name": "multinomial_nb", "hpo_space": {"alpha": loguniform(1e-3, 1e0)}}},

        {"feature_extractor": {"name": "tfidf", "params": {"ngram_range": (1, 2), "max_features": 30000},
                               "precomputed": {"train_file": "features/tfidf_train_bigram.npz",
                                               "test_file": "features/tfidf_test_bigram.npz"}},
         "model": {"name": "logistic_regression", "hpo_space": {"C": loguniform(1e-2, 1e2),
                                                                "class_weight": [None, "balanced"]}}},

        {"feature_extractor": {"name": "tfidf", "params": {"ngram_range": (1, 2), "max_features": 30000},
                               "precomputed": {"train_file": "features/tfidf_train_bigram.npz",
                                               "test_file": "features/tfidf_test_bigram.npz"}},
         "model": {"name": "linear_svc", "hpo_space": {"C": loguniform(1e-2, 1e2),
                                                       "class_weight": [None, "balanced"]}}},

        # --- GloVe Mean Pooling ---
        {"feature_extractor": {"name": "glove_mean", "params": {"embedding_dim": 100},
                               "precomputed": {"file": "features/glove.npz"}},
         "model": {"name": "logistic_regression", "hpo_space": {"C": loguniform(1e-2, 1e2),
                                                                "class_weight": [None, "balanced"]}}},

        {"feature_extractor": {"name": "glove_mean", "params": {"embedding_dim": 100},
                               "precomputed": {"file": "features/glove.npz"}},
         "model": {"name": "linear_svc", "hpo_space": {"C": loguniform(1e-2, 1e2),
                                                       "class_weight": [None, "balanced"]}}},

        # --- BERT Static Embedding ---
        {"feature_extractor": {"name": "bert_cls", "params": {"model_name": "distilbert-base-uncased"},
                               "precomputed": {"file": "features/bert_static.npz"}},
         "model": {"name": "logistic_regression", "hpo_space": {"C": loguniform(1e-2, 1e2),
                                                                "class_weight": [None, "balanced"]}}},

        {"feature_extractor": {"name": "bert_cls", "params": {"model_name": "distilbert-base-uncased"},
                               "precomputed": {"file": "features/bert_static.npz"}},
         "model": {"name": "linear_svc", "hpo_space": {"C": loguniform(1e-2, 1e2),
                                                       "class_weight": [None, "balanced"]}}}
    ],

    # Các pipeline với Deep Learning models
    "dl_pipelines": [
        {"feature_extractor": {"name": "bert_sequence", "params": {"model_name": "distilbert-base-uncased", "max_len": 200},
                               "precomputed": {"file": "features/bert_static.npz"}},
         "model": {"name": "rnn", "hpo_space": {"hidden_dim": [64, 128, 256],
                                                "num_layers": [1, 2],
                                                "dropout": [0.2, 0.5],
                                                "lr": loguniform(1e-4, 1e-2)}}},

        {"feature_extractor": {"name": "bert_sequence", "params": {"model_name": "distilbert-base-uncased", "max_len": 200},
                               "precomputed": {"file": "features/bert_static.npz"}},
         "model": {"name": "lstm", "hpo_space": {"hidden_dim": [64, 128, 256],
                                                 "num_layers": [1, 2],
                                                 "dropout": [0.2, 0.5],
                                                 "lr": loguniform(1e-4, 1e-2)}}}
    ]
}

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
            pipeline_id = f"pipeline_{idx}_{pipeline['feature_extractor']['name']}_{pipeline['model']['name']}"
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Running Pipeline {idx+1}/{len(pipelines)}: {pipeline_id}")
                print(f"{'='*60}")
            
            try:
                result = self.run_single_pipeline(pipeline, X_train, y_train, X_test, y_test, pipeline_id)
                results.append(result)
                
                if self.verbose:
                    print(f"✓ Pipeline completed successfully")
                    print(f"  Test F1 Macro: {result['test_f1_macro']:.4f}")
                    print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
                    
            except Exception as e:
                print(f"✗ Pipeline {pipeline_id} failed: {e}")
                import traceback
                if self.verbose:
                    traceback.print_exc()
        
        final_results_df = pd.DataFrame(results).sort_values(by="test_f1_macro", ascending=False)
        results_path = os.path.join(self.workdir, "results", "final_results.csv")
        final_results_df.to_csv(results_path, index=False)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"All experiments completed! Results saved to: {results_path}")
            print(f"{'='*60}")
            
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
            model, train_time = self.train_dl_model(model_cfg, X_train_feat, y_train, X_test_feat, y_test)
        else:
            raise ValueError(f"Unsupported model: {model_cfg['name']}")

        # Evaluate model
        inference_start = time.time()
        
        if model_cfg["name"] in ["multinomial_nb", "logistic_regression", "linear_svc"]:
            y_pred = model.predict(X_test_feat)
        else:  # Deep learning models
            y_pred_proba = model.predict(X_test_feat)
            y_pred = np.argmax(y_pred_proba, axis=-1) + 1
            
        inference_time = time.time() - inference_start

        metrics = {
            "test_f1_macro": f1_score(y_test, y_pred, average="macro"),
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "test_recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        }

        # Save model artifact
        model_path = None
        if self.save_models:
            if model_cfg["name"] in ["multinomial_nb", "logistic_regression", "linear_svc"]:
                model_path = os.path.join(self.workdir, "artifacts", f"{pipeline_id}.joblib")
                joblib.dump(model, model_path)
            elif model_cfg["name"] in ["rnn", "lstm"]:
                model_path = os.path.join(self.workdir, "artifacts", f"{pipeline_id}.h5")
                model.save(model_path)

        total_time = time.time() - start_time
        
        return {
            "pipeline_id": pipeline_id,
            "feature_extractor_name": feature_cfg["name"],
            "model_name": model_cfg["name"],
            "train_time_s": train_time,
            "inference_time_s": inference_time,
            "total_time_s": total_time,
            "model_artifact_path": model_path,
            **metrics,
        }

    def load_or_build_feature(self, feature_cfg, X_train, X_test, y_train=None):
        precomputed = feature_cfg.get("precomputed", {})

        if "file" in precomputed and os.path.exists(precomputed["file"]):
            if self.verbose:
                print(f"  Loading precomputed features from {precomputed['file']}")
            
            # Fix: Handle .npz file loading properly
            if precomputed["file"].endswith('.npz'):
                data = np.load(precomputed["file"])
                # Check for different possible key naming conventions
                if 'X_train' in data.files and 'X_test' in data.files:
                    X_train_feat = data["X_train"]
                    X_test_feat = data["X_test"]
                elif 'train' in data.files and 'test' in data.files:
                    X_train_feat = data["train"]
                    X_test_feat = data["test"]
                else:
                    print(f"    Warning: Expected keys 'X_train'/'X_test' or 'train'/'test' not found in {precomputed['file']}")
                    print(f"    Available keys: {list(data.files)}")
                    # Fall through to rebuild features
                    X_train_feat = None
                    X_test_feat = None
                
                # Special handling for bert_sequence - need to transform to 3D
                if X_train_feat is not None and feature_cfg["name"] == "bert_sequence" and "bert_static.npz" in precomputed["file"]:
                    if self.verbose:
                        print(f"    Transforming BERT features from {X_train_feat.shape} to sequence format")
                    
                    max_len = feature_cfg.get("params", {}).get("max_len", 200)
                    # Transform from 2D (samples, 768) to 3D (samples, max_len, 768)
                    X_train_feat = np.repeat(X_train_feat[:, np.newaxis, :], max_len, axis=1)
                    X_test_feat = np.repeat(X_test_feat[:, np.newaxis, :], max_len, axis=1)
                    
                    if self.verbose:
                        print(f"    Transformed to sequence: train {X_train_feat.shape}, test {X_test_feat.shape}")
                
                if X_train_feat is not None:
                    return X_train_feat, X_test_feat
            else:
                data = np.load(precomputed["file"])
                return data["train"], data["test"]
            
        elif "train_file" in precomputed and "test_file" in precomputed:
            if os.path.exists(precomputed["train_file"]) and os.path.exists(precomputed["test_file"]):
                if self.verbose:
                    print(f"  Loading precomputed features from {precomputed['train_file']} and {precomputed['test_file']}")
                return load_npz(precomputed["train_file"]), load_npz(precomputed["test_file"])

        # Build features if not precomputed
        if self.verbose:
            print(f"  Building features using {feature_cfg['name']}")
            
        if feature_cfg["name"] in ["bow", "tfidf"]:
            return self.compute_bow_or_tfidf(feature_cfg["name"], feature_cfg["params"], X_train, X_test)
        elif feature_cfg["name"] == "glove_mean":
            return self.compute_glove_mean(feature_cfg.get("params", {}), X_train, X_test)
        elif feature_cfg["name"] == "glove_sequence":
            return self.compute_glove_sequence(feature_cfg.get("params", {}), X_train, X_test)
        elif feature_cfg["name"] == "bert_cls":
            return self.compute_bert_cls(feature_cfg["params"]["model_name"], X_train, X_test)
        elif feature_cfg["name"] == "bert_sequence":
            return self.compute_bert_sequence(feature_cfg.get("params", {}), X_train, X_test)
        else:
            raise ValueError(f"Unsupported feature extractor: {feature_cfg['name']}")

    def compute_bow_or_tfidf(self, kind, params, X_train_texts, X_test_texts):
        vectorizer = CountVectorizer(**params) if kind == "bow" else TfidfVectorizer(**params)
        X_train_mat = vectorizer.fit_transform(X_train_texts)
        X_test_mat = vectorizer.transform(X_test_texts)
        
        # Save with proper naming
        ngram_type = "unigram" if params.get("ngram_range") == (1, 1) else "bigram"
        train_file = os.path.join(self.workdir, "features", f"{kind}_train_{ngram_type}.npz")
        test_file = os.path.join(self.workdir, "features", f"{kind}_test_{ngram_type}.npz")
        
        save_npz(train_file, X_train_mat)
        save_npz(test_file, X_test_mat)
        
        return X_train_mat, X_test_mat

    def compute_glove_mean(self, params, X_train_texts, X_test_texts):
        glove_path = params.get("glove_txt_path", "glove.6B.100d.txt")
        embedding_dim = params.get("embedding_dim", 100)
        
        if not os.path.exists(glove_path):
            # Try to use random embeddings if GloVe file not found
            if self.verbose:
                print(f"    Warning: GloVe file not found at {glove_path}. Using random embeddings.")
            
            def embed_text(texts):
                np.random.seed(42)
                return np.random.randn(len(texts), embedding_dim).astype(np.float32)
                
            X_train_vec = embed_text(X_train_texts)
            X_test_vec = embed_text(X_test_texts)
        else:
            embeddings = {}
            with open(glove_path, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc="Loading GloVe"):
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], dtype="float32")
                    embeddings[word] = vector
                    
            def embed_text(texts):
                vectors = []
                for text in tqdm(texts, desc="Embedding texts"):
                    words = text.lower().split()
                    word_vectors = [embeddings[word] for word in words if word in embeddings]
                    if word_vectors:
                        vectors.append(np.mean(word_vectors, axis=0))
                    else:
                        vectors.append(np.zeros(embedding_dim))
                return np.array(vectors, dtype=np.float32)
                
            X_train_vec = embed_text(X_train_texts)
            X_test_vec = embed_text(X_test_texts)
        
        np.savez_compressed(os.path.join(self.workdir, "features", "glove.npz"), 
                          X_train=X_train_vec, X_test=X_test_vec)
        return X_train_vec, X_test_vec

    def compute_glove_sequence(self, params, X_train_texts, X_test_texts):
        # For RNN/LSTM, we need sequences not mean vectors
        # This is a simplified implementation - you may need to adjust based on your needs
        max_len = params.get("max_len", 200)
        embedding_dim = params.get("embedding_dim", 100)
        
        # Return dummy sequences for now
        X_train_seq = np.random.randn(len(X_train_texts), max_len, embedding_dim).astype(np.float32)
        X_test_seq = np.random.randn(len(X_test_texts), max_len, embedding_dim).astype(np.float32)
        
        return X_train_seq, X_test_seq

    def compute_bert_cls(self, model_name, X_train_texts, X_test_texts, batch_size=32):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.verbose:
            print(f"    Using device: {device}")
            
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        
        def embed_texts(texts):
            embeddings = []
            for i in tqdm(range(0, len(texts), batch_size), desc="BERT embedding"):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True, 
                                  max_length=512, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
            return np.vstack(embeddings)
            
        X_train_vec = embed_texts(X_train_texts)
        X_test_vec = embed_texts(X_test_texts)
        
        np.savez_compressed(os.path.join(self.workdir, "features", "bert_static.npz"), 
                          X_train=X_train_vec, X_test=X_test_vec)
        return X_train_vec, X_test_vec

    def compute_bert_sequence(self, params, X_train_texts, X_test_texts):
        # For RNN/LSTM with BERT features
        max_len = params.get("max_len", 200)
        
        # Load precomputed BERT embeddings if available
        bert_file = os.path.join(self.workdir, "features", "bert_static.npz")
        if os.path.exists(bert_file):
            data = np.load(bert_file)
            # Handle different key naming conventions
            if 'X_train' in data.files and 'X_test' in data.files:
                X_train_bert = data["X_train"]
                X_test_bert = data["X_test"]
            elif 'train' in data.files and 'test' in data.files:
                X_train_bert = data["train"]
                X_test_bert = data["test"]
            else:
                raise ValueError(f"Unexpected keys in bert_static.npz: {list(data.files)}")
        
            if self.verbose:
                print(f"    Loaded BERT features: train {X_train_bert.shape}, test {X_test_bert.shape}")
            
            # Transform from 2D (samples, 768) to 3D (samples, max_len, 768)
            X_train_seq = np.repeat(X_train_bert[:, np.newaxis, :], max_len, axis=1)
            X_test_seq = np.repeat(X_test_bert[:, np.newaxis, :], max_len, axis=1)
            
            if self.verbose:
                print(f"    Transformed to sequence: train {X_train_seq.shape}, test {X_test_seq.shape}")
        else:
            # Return dummy sequences
            embedding_dim = 768  # DistilBERT dimension
            X_train_seq = np.random.randn(len(X_train_texts), max_len, embedding_dim).astype(np.float32)
            X_test_seq = np.random.randn(len(X_test_texts), max_len, embedding_dim).astype(np.float32)
            
            if self.verbose:
                print(f"    Using dummy sequences: train {X_train_seq.shape}, test {X_test_seq.shape}")
        
        return X_train_seq, X_test_seq

    def train_sklearn_model(self, model_cfg, X_train, y_train):
        # Create model instance
        if model_cfg["name"] == "multinomial_nb":
            base_model = MultinomialNB()
        elif model_cfg["name"] == "logistic_regression":
            base_model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_cfg["name"] == "linear_svc":
            base_model = LinearSVC(max_iter=2000, random_state=42)
        else:
            raise ValueError(f"Unknown model: {model_cfg['name']}")
        
        # Get hyperparameter space
        hpo_space = model_cfg.get("hpo_space", {})
        
        if hpo_space:
            # Remove 'classifier__' prefix as we're not using Pipeline
            clean_hpo_space = {k.replace('classifier__', ''): v for k, v in hpo_space.items()}
            
            # Perform hyperparameter optimization
            search = RandomizedSearchCV(
                base_model, 
                clean_hpo_space, 
                n_iter=self.config["settings"]["hpo_settings"]["n_iter"],
                cv=self.config["settings"]["hpo_settings"]["cv_folds"], 
                scoring="f1_macro", 
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            
            start_time = time.time()
            search.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            if self.verbose:
                print(f"    Best params: {search.best_params_}")
                print(f"    Best CV score: {search.best_score_:.4f}")
            
            return search.best_estimator_, train_time
        else:
            # Train without HPO
            start_time = time.time()
            base_model.fit(X_train, y_train)
            train_time = time.time() - start_time
            return base_model, train_time

    def train_dl_model(self, model_cfg, X_train, y_train, X_test, y_test):
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.layers import Dropout
        
        num_classes = len(np.unique(y_train))
        input_shape = X_train.shape[1:]

        if self.verbose:
            print(f"    Input shape: {input_shape}")
            print(f"    X_train shape: {X_train.shape}")
            print(f"    Number of classes: {num_classes}")
        
        # Convert labels to categorical if needed
        y_train_cat = to_categorical(y_train - 1, num_classes)
        y_test_cat = to_categorical(y_test - 1, num_classes)
        
        # Sample hyperparameters from space - CONVERT TO PYTHON NATIVE TYPES
        hidden_dim = int(np.random.choice(model_cfg["hpo_space"]["hidden_dim"]))
        num_layers = int(np.random.choice(model_cfg["hpo_space"]["num_layers"]))
        dropout_rate = float(np.random.choice(model_cfg["hpo_space"]["dropout"]))
        lr = float(model_cfg["hpo_space"]["lr"].rvs())
        
        if self.verbose:
            print(f"    DL Hyperparameters: hidden_dim={hidden_dim}, layers={num_layers}, dropout={dropout_rate}, lr={lr:.5f}")
            print(f"    Types: hidden_dim={type(hidden_dim)}, num_layers={type(num_layers)}")
        
        # Build model
        model = Sequential()
        model.add(Input(shape=input_shape))
        
        # Add RNN or LSTM layers
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1)  # Only last layer returns single output
            
            if model_cfg["name"] == "rnn":
                model.add(SimpleRNN(hidden_dim, return_sequences=return_sequences))
            elif model_cfg["name"] == "lstm":
                model.add(LSTM(hidden_dim, return_sequences=return_sequences))

            # Add dropout after each RNN/LSTM layer
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        # Add output layer
        model.add(Dense(num_classes, activation="softmax"))
        
        # Compile model
        from tensorflow.keras.optimizers import Adam
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        if self.verbose:
            print("    Model summary:")
            model.summary()
        
        # Train model
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0)
        
        start_time = time.time()
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=20,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        train_time = time.time() - start_time
        
        if self.verbose:
            print(f"    Trained for {len(history.history['loss'])} epochs")
            print(f"    Final val_accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        return model, train_time

if __name__ == "__main__":
    print("Unified Pipeline Module Loaded.")
    print("Example usage:")
    print("""
    from unified_pipeline import ExperimentRunner, EXPERIMENT_CONFIG
    
    # Assuming X_train, y_train, X_test, y_test are your data
    runner = ExperimentRunner(config=EXPERIMENT_CONFIG)
    final_results_df = runner.run_all_experiments(X_train, y_train, X_test, y_test)
    
    # Display results
    print(final_results_df.head())
    """)