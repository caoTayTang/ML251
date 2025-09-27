# ==============================================================================
# File: modules/unified_pipeline.py
# Chứa class ExperimentRunner đa năng để chạy tất cả các thử nghiệm.
# ==============================================================================

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- Helper Classes & Functions ---

class PytorchSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class TransformerDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}
    
class SequenceClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, model_type='lstm', n_layers=2, dropout=0.5):
        super().__init__()
        RecurrentLayer = nn.LSTM if model_type == 'lstm' else nn.RNN
        self.recurrent = RecurrentLayer(embedding_dim, hidden_dim, num_layers=n_layers,
                                        bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, text_vectors):
        packed_output, (hidden, cell) = self.recurrent(text_vectors)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

# --- Main Class: ExperimentRunner ---

class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.results = []
        self.settings = config.get("settings", {})
        self.cache_path = self.settings.get("cache_path", "features")
        os.makedirs(self.cache_path, exist_ok=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"ExperimentRunner khởi tạo trên thiết bị: {self.device}")

    # --- Core Dispatcher Method ---
    
    def run_all_experiments(self, X_train, y_train, X_test, y_test):
        feature_extractors_config = self.config.get("feature_extractors", {})
        models_config = self.config.get("models", {})
        finetune_config = self.config.get("fine_tuning_models", {})

        # 1. Chạy các pipeline kết hợp (Feature Extractor + Model)
        for extractor_name, extractor_params in feature_extractors_config.items():
            print(f"\n--- [Extractor] Bắt đầu xử lý đặc trưng cho: {extractor_name} ---")
            X_train_feat, X_test_feat = self._get_features(extractor_name, X_train, X_test, extractor_params)

            for model_name, model_params in models_config.items():
                if not self._is_compatible(extractor_name, model_name):
                    print(f"  > [Skip] Bỏ qua kết hợp không tương thích: {extractor_name} + {model_name}")
                    continue

                print(f"  > [Model] Bắt đầu huấn luyện: {model_name} trên {extractor_name}")
                if model_name in ["logistic_regression", "svm", "naive_bayes"]:
                    self._run_sklearn_model(model_name, X_train_feat, y_train, X_test_feat, y_test, extractor_name, model_params)
                elif model_name in ["lstm", "rnn"]:
                    self._run_pytorch_sequence_model(model_name, X_train_feat, y_train, X_test_feat, y_test, extractor_name, model_params)
        
        # 2. Chạy pipeline Fine-tuning độc lập
        for model_name, model_params in finetune_config.items():
            print(f"\n--- [Fine-Tune] Bắt đầu pipeline: {model_name} ---")
            self._run_fine_tuning(model_name, X_train, y_train, X_test, y_test, model_params)
            
        return pd.DataFrame(self.results)

    def _is_compatible(self, extractor_name, model_name):
        if model_name == "naive_bayes" and extractor_name not in ["tfidf", "bow"]: return False
        if model_name in ["lstm", "rnn"] and extractor_name not in ["glove", "word2vec"]: return False
        if model_name in ["logistic_regression", "svm"] and extractor_name in ["glove", "word2vec"]: return False
        return True

    # --- Feature Extraction & Caching ---

    def _get_features(self, extractor_name, X_train, X_test, params):
        feature_file_base = os.path.join(self.cache_path, f"{extractor_name}")
        
        if self.settings.get("use_cache") and os.path.exists(f"{feature_file_base}_train.npz"):
            print(f"  > Tải features từ cache: {feature_file_base}_*.npz")
            X_train_feat = load_npz(f"{feature_file_base}_train.npz")
            X_test_feat = load_npz(f"{feature_file_base}_test.npz")
            return X_train_feat, X_test_feat

        if self.settings.get("use_cache") and os.path.exists(f"{feature_file_base}.npz"):
            print(f"  > Tải features từ cache: {feature_file_base}.npz")
            data = np.load(f"{feature_file_base}.npz", allow_pickle=True)
            return data['train'], data['test']
            
        print(f"  > Tính toán và lưu features cho: {extractor_name}")
        if extractor_name in ["tfidf", "bow"]:
            vec_class = TfidfVectorizer if extractor_name == "tfidf" else CountVectorizer
            vec = vec_class(**params.get("vectorizer_params", {}))
            X_train_feat = vec.fit_transform(X_train)
            X_test_feat = vec.transform(X_test)
            save_npz(f"{feature_file_base}_train.npz", X_train_feat)
            save_npz(f"{feature_file_base}_test.npz", X_test_feat)
        elif extractor_name == "bert_static":
            X_train_feat, X_test_feat = self._extract_transformer_embeddings(params['model'], X_train, X_test)
            np.savez_compressed(f"{feature_file_base}.npz", train=X_train_feat, test=X_test_feat)
        elif extractor_name == "glove":
            embedding_model = self._load_glove_embeddings(params['path'])
            X_train_feat = self._texts_to_sequences(X_train, embedding_model)
            X_test_feat = self._texts_to_sequences(X_test, embedding_model)
            np.savez_compressed(f"{feature_file_base}.npz", train=X_train_feat, test=X_test_feat)
        else:
            raise ValueError(f"Feature extractor '{extractor_name}' không được hỗ trợ.")
        return X_train_feat, X_test_feat

    # --- Model Runner Helpers ---

    def _run_sklearn_model(self, model_name, X_train_vec, y_train, X_test_vec, y_test, extractor_name, params):
        model_map = {"logistic_regression": LogisticRegression, "svm": LinearSVC, "naive_bayes": MultinomialNB}
        model = model_map[model_name](**params.get("model_params", {}))
        
        start_time = time.time()
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        duration = time.time() - start_time
        
        self.results.append({
            "type": "sklearn", "extractor": extractor_name, "model": model_name,
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
            "accuracy": accuracy_score(y_test, y_pred),
            "time_seconds": duration
        })
        print(f"    -> Hoàn thành trong {duration:.2f}s. F1-Score: {self.results[-1]['f1_score']:.4f}")

    def _run_pytorch_sequence_model(self, model_name, X_train_seq, y_train, X_test_seq, y_test, extractor_name, params):
        embedding_dim = X_train_seq.shape[2]
        num_labels = len(y_train.unique())
        y_train_mapped = y_train.values - 1
        y_test_mapped = y_test.values - 1
        
        train_dataset = PytorchSequenceDataset(X_train_seq, y_train_mapped)
        test_dataset = PytorchSequenceDataset(X_test_seq, y_test_mapped)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=params.get('batch_size', 64))
        test_loader = DataLoader(test_dataset, batch_size=params.get('batch_size', 128))

        model = SequenceClassifier(embedding_dim, **params.get("model_params", {}), output_dim=num_labels, model_type=model_name).to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss().to(self.device)
        
        start_time = time.time()
        model.train()
        for epoch in range(params.get('epochs', 3)):
            for texts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                texts, labels = texts.to(self.device), labels.to(self.device)
                optimizer.zero_grad(); output = model(texts); loss = criterion(output, labels); loss.backward(); optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                all_preds.extend(model(texts).argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        duration = time.time() - start_time

        self.results.append({
            "type": "sequence", "extractor": extractor_name, "model": model_name,
            "f1_score": f1_score(all_labels, all_preds, average='weighted'),
            "accuracy": accuracy_score(all_labels, all_preds), "time_seconds": duration
        })
        print(f"    -> Hoàn thành trong {duration:.2f}s. F1-Score: {self.results[-1]['f1_score']:.4f}")

    def _run_fine_tuning(self, model_name, X_train, y_train, X_test, y_test, params):
        model_save_path = os.path.join(self.settings.get("model_save_path", "models"), model_name)
        os.makedirs(model_save_path, exist_ok=True)
        
        y_train_mapped, y_test_mapped = y_train - 1, y_test - 1
        num_labels = len(y_train.unique())
        
        start_time = time.time()
        if self.settings.get("use_cache") and os.path.exists(os.path.join(model_save_path, "config.json")):
            print(f"  > Tải mô hình fine-tuned từ cache: {model_save_path}")
            model = AutoModelForSequenceClassification.from_pretrained(model_save_path).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            trainer_mode = "evaluate"
        else:
            print(f"  > Huấn luyện mô hình fine-tuning từ đầu: {model_name}")
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            trainer_mode = "train"

        train_dataset = TransformerDataset(tokenizer(X_train.tolist(), **params.get("tokenizer_params", {})), y_train_mapped.tolist())
        test_dataset = TransformerDataset(tokenizer(X_test.tolist(), **params.get("tokenizer_params", {})), y_test_mapped.tolist())
        
        training_args_dict = params.get("training_params", {})
        training_args_dict['logging_dir'] = os.path.join(self.settings.get("log_path", "logs"), model_name)
        training_args_dict['output_dir'] = os.path.join(self.settings.get("checkpoint_path", "checkpoints"), model_name)
        training_args = TrainingArguments(**training_args_dict)

        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=compute_metrics)
        
        if trainer_mode == "train":
            trainer.train()
            trainer.save_model(model_save_path)
            print(f"  > Đã lưu model vào: {model_save_path}")

        eval_results = trainer.evaluate()
        duration = time.time() - start_time
        
        self.results.append({
            "type": "fine-tune", "extractor": "N/A", "model": model_name,
            "f1_score": eval_results['eval_f1'], "accuracy": eval_results['eval_accuracy'],
            "time_seconds": duration
        })
        print(f"    -> Hoàn thành trong {duration:.2f}s. F1-Score: {self.results[-1]['f1_score']:.4f}")

    # --- Utility methods for feature processing ---

    def _extract_transformer_embeddings(self, model_name, X_train, X_test):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        def get_embeddings(texts):
            all_embeddings = []
            for i in tqdm(range(0, len(texts), 32), desc="  > Extracting Embeddings", leave=False):
                batch = texts[i:i+32].tolist()
                inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256).to(self.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                all_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
            return np.vstack(all_embeddings)
        return get_embeddings(X_train), get_embeddings(X_test)
        
    def _load_glove_embeddings(self, path):
        print(f"  > Đang tải GloVe embeddings từ {path}...")
        embeddings_dict = {}
        with open(path, 'r', encoding="utf-8") as f:
            for line in tqdm(f, desc="  > Loading GloVe file"):
                values = line.split()
                word = values[0]
                embeddings_dict[word] = np.asarray(values[1:], "float32")
        return embeddings_dict

    def _texts_to_sequences(self, texts, embedding_model, max_len=128):
        embedding_dim = len(next(iter(embedding_model.values())))
        sequences = []
        for text in tqdm(texts, desc="  > Processing texts to sequences", leave=False):
            tokens = nltk.word_tokenize(text.lower())
            vectors = [embedding_model.get(token, np.zeros(embedding_dim)) for token in tokens[:max_len]]
            if len(vectors) < max_len:
                vectors.extend([np.zeros(embedding_dim)] * (max_len - len(vectors)))
            sequences.append(vectors)
        return np.array(sequences, dtype=np.float32)