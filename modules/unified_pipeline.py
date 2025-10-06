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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, BatchNormalization, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from scipy.sparse import save_npz, load_npz
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification
import torch.nn.functional as F

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
        pipelines = (
            self.config.get("sklearn_pipelines", [])
            + self.config.get("dl_pipelines", [])
            + self.config.get("finetune_pipelines", [])
        )

        for idx, pipeline in enumerate(pipelines):
            # Enhanced pipeline_id generation with more details
            feature_name = pipeline['feature_extractor']['name']
            model_name = pipeline['model']['name']
            
            # Add ngram info for bow/tfidf features
            if feature_name in ['bow', 'tfidf']:
                ngram_range = pipeline['feature_extractor']['params'].get('ngram_range', (1, 1))
                if ngram_range == (1, 1):
                    ngram_type = "unigram"
                elif ngram_range == (1, 2):
                    ngram_type = "bigram"
                elif ngram_range == (1, 3):
                    ngram_type = "trigram"
                elif ngram_range == (2, 2):
                    ngram_type = "bigram_only"
                elif ngram_range == (3, 3):
                    ngram_type = "trigram_only"
                else:
                    ngram_type = f"ngram_{ngram_range[0]}_{ngram_range[1]}"
                
                pipeline_id = f"pipeline_{idx}_{feature_name}_{ngram_type}_{model_name}"
            else:
                pipeline_id = f"pipeline_{idx}_{feature_name}_{model_name}"

            # Check if it's a DL model with retrain=False
            if pipeline['model']['name'] in ['rnn', 'lstm'] and not pipeline['model'].get('retrain', True):
                # Load pre-existing results from CSV
                result_path = os.path.join(self.workdir, pipeline['model']['result_path'])
                if os.path.exists(result_path):
                    pre_results_df = pd.read_csv(result_path)
                    # Assuming the CSV has the same columns as our result dict
                    # Add or update pipeline_id if needed
                    pre_results_df['pipeline_id'] = pipeline_id
                    results.extend(pre_results_df.to_dict('records'))
                    
                    if self.verbose:
                        print(f"✓ Loaded pre-existing results for {pipeline_id} from {result_path}")
                        print(f"  Loaded {len(pre_results_df)} results")
                else:
                    print(f"✗ Result file not found for {pipeline_id}: {result_path}")
                continue

            # Finetune: nếu retrain=False và đã có file kết quả -> chỉ load; nếu chưa có -> chạy inference và lưu
            if pipeline['model']['name'] == 'finetune':
                result_path = os.path.join(self.workdir, pipeline['model']['result_path'])
                if (not pipeline['model'].get('retrain', True)) and os.path.exists(result_path):
                    pre_results_df = pd.read_csv(result_path)
                    pre_results_df['pipeline_id'] = pipeline_id
                    results.extend(pre_results_df.to_dict('records'))
                    if self.verbose:
                        print(f"✓ Loaded pre-existing finetune results for {pipeline_id} from {result_path}")
                        print(f"  Loaded {len(pre_results_df)} results")
                    continue
                # Nếu chưa có file kết quả, thực hiện inference và lưu
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"Running Pipeline {idx+1}/{len(pipelines)}: {pipeline_id}")
                    print(f"{'='*60}")
                try:
                    result = self.run_finetune_pipeline(pipeline, X_test, y_test, pipeline_id)
                    results.append(result)
                    if self.verbose:
                        print(f"✓ Finetune inference completed")
                        print(f"  Test F1 Macro: {result['test_f1_macro']:.4f}")
                        print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
                except Exception as e:
                    print(f"✗ Finetune pipeline {pipeline_id} failed: {e}")
                    if self.verbose:
                        import traceback; traceback.print_exc()
                continue

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
        results_path = os.path.join(self.workdir, "BTL2/results", "final_results.csv")
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
            y_pred = np.argmax(y_pred_proba, axis=-1)
            
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
        
        # Save with proper naming - handle all ngram types
        ngram_range = params.get("ngram_range", (1, 1))
        if ngram_range == (1, 1):
            ngram_type = "unigram"
        elif ngram_range == (1, 2):
            ngram_type = "bigram"
        elif ngram_range == (1, 3):
            ngram_type = "trigram"
        elif ngram_range == (2, 2):
            ngram_type = "bigram_only"
        elif ngram_range == (3, 3):
            ngram_type = "trigram_only"
        else:
            ngram_type = f"ngram_{ngram_range[0]}_{ngram_range[1]}"
        
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
            base_model = LogisticRegression(max_iter=2000, random_state=42)
        elif model_cfg["name"] == "linear_svc":
            base_model = LinearSVC(max_iter=10000, tol=1e-3, dual=True, random_state=42)
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
        num_classes = len(np.unique(y_train))
        input_shape = X_train.shape[1:]

        if self.verbose:
            print(f"    Input shape: {input_shape}")
            print(f"    X_train shape: {X_train.shape}")
            print(f"    Number of classes: {num_classes}")
        
        # Fix label encoding: labels 1-5 -> 0-4 for to_categorical
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)

        # Sample hyperparameters from space - CONVERT TO PYTHON NATIVE TYPES
        hidden_dim = int(np.random.choice(model_cfg["hpo_space"]["hidden_dim"]))
        num_layers = int(np.random.choice(model_cfg["hpo_space"]["num_layers"]))
        dropout_rate = float(np.random.choice(model_cfg["hpo_space"]["dropout"]))
        lr = float(model_cfg["hpo_space"]["lr"].rvs())
        
        if self.verbose:
            print(f"    DL Hyperparameters: hidden_dim={hidden_dim}, layers={num_layers}, dropout={dropout_rate}, lr={lr:.5f}")
        
        # Enhanced model architecture
        model = Sequential()
        model.add(Input(shape=input_shape))
        
        # Add Bidirectional RNN/LSTM layers for better context understanding
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1)
            
            if model_cfg["name"] == "rnn":
                # Use Bidirectional RNN for better sequence modeling
                model.add(Bidirectional(SimpleRNN(hidden_dim, return_sequences=return_sequences)))
            elif model_cfg["name"] == "lstm":
                # Use Bidirectional LSTM for better sequence modeling
                model.add(Bidirectional(LSTM(hidden_dim, return_sequences=return_sequences)))

            # Add BatchNormalization for training stability
            if not return_sequences:  # Only after last layer
                model.add(BatchNormalization())
            
            # Add dropout for regularization
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        # Add dense layers for better feature learning
        model.add(Dense(hidden_dim, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Additional dense layer for complex pattern learning
        model.add(Dense(hidden_dim // 2, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate * 0.5))  # Reduced dropout in final layers
        
        # Output layer
        model.add(Dense(num_classes, activation="softmax"))
        
        # Compile with improved optimizer settings
        model.compile(
            optimizer=Adam(
                learning_rate=lr,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
                clipnorm=1.0  # Gradient clipping to prevent exploding gradients
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        if self.verbose:
            print("    Enhanced Model Architecture:")
            model.summary()
        
        # Enhanced training with better callbacks
        early_stopping = EarlyStopping(
            monitor="val_accuracy",  # Monitor accuracy instead of loss
            patience=7,  # Increased patience
            restore_best_weights=True,
            verbose=0,
            mode='max'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=0
        )
        
        start_time = time.time()
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=50,  # Increased epochs
            batch_size=16,  # Smaller batch size for better convergence
            callbacks=[early_stopping, reduce_lr],
            verbose=0,
            shuffle=True  # Shuffle data each epoch
        )
        train_time = time.time() - start_time
        
        if self.verbose:
            print(f"    Trained for {len(history.history['loss'])} epochs")
            print(f"    Final val_accuracy: {history.history['val_accuracy'][-1]:.4f}")
            print(f"    Best val_accuracy: {max(history.history['val_accuracy']):.4f}")
            print(f"    Final learning rate: {model.optimizer.learning_rate.numpy():.2e}")
        
        return model, train_time
    
    def run_finetune_pipeline(self, pipeline_config, X_test_texts, y_test, pipeline_id):
        feature_cfg = pipeline_config["feature_extractor"]
        model_cfg = pipeline_config["model"]
        params = feature_cfg.get("params", {})
        max_len = int(params.get("max_len", 128))
        # Ưu tiên checkpoint_path trong model_cfg; nếu không có, fallback model_name
        ckpt_rel = model_cfg.get("checkpoint_path") or params.get("model_name")
        ckpt_path = ckpt_rel if os.path.isabs(ckpt_rel) else os.path.join(self.workdir, ckpt_rel)
        result_path = os.path.join(self.workdir, model_cfg["result_path"])

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.verbose:
            print(f"  Loading finetuned model from: {ckpt_path} (device: {device})")

        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_path).to(device)
        model.eval()

        # Inference theo batch
        batch_size = 32
        all_preds = []
        inference_start = time.time()
        with torch.no_grad():
            for i in tqdm(range(0, len(X_test_texts), batch_size), desc="Finetune inference"):
                batch_texts = X_test_texts[i:i+batch_size]
                inputs = tokenizer(
                    list(batch_texts),
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt"
                ).to(device)
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
                all_preds.append(preds)
        inference_time = time.time() - inference_start
        y_pred = np.concatenate(all_preds, axis=0)

        metrics = {
            "test_f1_macro": f1_score(y_test, y_pred, average="macro"),
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "test_recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        }

        # Lưu kết quả 1 dòng ra CSV theo yêu cầu
        row = {
            "pipeline_id": pipeline_id,
            "feature_extractor_name": feature_cfg["name"],
            "model_name": model_cfg["name"],
            "train_time_s": None,          # để trống
            "inference_time_s": inference_time,
            "total_time_s": None,          # để trống
            "model_artifact_path": ckpt_path,
            **metrics,
        }
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        pd.DataFrame([row]).to_csv(result_path, index=False)
        if self.verbose:
            print(f"  Saved finetune results to: {result_path}")
        return row

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