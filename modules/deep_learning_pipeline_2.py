import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import time
import os
import gensim
import nltk

# ==============================================================================
# CLASS 1: TransformerPipelineRunner (Tên mới và được nâng cấp)
# ==============================================================================

# --- Helper classes/functions cho Transformers (giữ nguyên) ---
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels): self.encodings, self.labels = encodings, labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]); return item
    def __len__(self): return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids; preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted'); acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}

class TransformerPipelineRunner:
    def __init__(self, config):
        self.config = config
        self.results = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Transformer Runner sử dụng thiết bị: {self.device}")

    def _extract_static_embeddings(self, texts, model_name, batch_size=32):
        print(f"  > Bắt đầu trích xuất static embeddings từ '{model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="  > Extracting Embeddings"):
            batch = texts[i:i+batch_size].tolist()
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)
        return np.vstack(all_embeddings)

    def run_static_embedding_experiment(self, X_train, y_train, X_test, y_test):
        model_name = self.config.get("embedding_models", [])[0]
        classifier_name = self.config.get("downstream_approaches", {}).get("static_embedding_with_classifier", [])[0]
        
        # --- NÂNG CẤP: LOGIC CACHING CHO STATIC EMBEDDINGS ---
        model_name_safe = model_name.replace('/', '_') # Tạo tên file an toàn
        feature_cache_path = f"ML251/features/{model_name_safe}_static_embeddings.npz"

        if os.path.exists(feature_cache_path):
            print(f"  > Tải static embeddings từ cache: {feature_cache_path}")
            data = np.load(feature_cache_path)
            train_embeddings, test_embeddings = data['train'], data['test']
        else:
            print("  > Không tìm thấy cache, bắt đầu trích xuất embeddings...")
            train_embeddings = self._extract_static_embeddings(X_train, model_name)
            test_embeddings = self._extract_static_embeddings(X_test, model_name)
            np.savez(feature_cache_path, train=train_embeddings, test=test_embeddings)
            print(f"  > Đã lưu embeddings vào cache: {feature_cache_path}")
        # --- KẾT THÚC NÂNG CẤP ---

        print(f"  > Bắt đầu huấn luyện mô hình classifier '{classifier_name}'...")
        start_time = time.time()
        classifier = LogisticRegression(max_iter=1000).fit(train_embeddings, y_train)
        y_pred = classifier.predict(test_embeddings)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        duration = time.time() - start_time
        
        self.results.append({
            "pipeline_type": "deep_learning_static", "embedding_model": model_name,
            "classifier": classifier_name, "accuracy": accuracy, "f1_score": f1,
            "training_time_seconds": duration
        })
        print(f"    -> Hoàn thành trong {duration:.2f} giây. F1-Score: {f1:.4f}")

    def run_finetuning_experiment(self, X_train, y_train, X_test, y_test):
        model_name = self.config.get("embedding_models", [])[0]
        model_name_safe = model_name.replace('/', '_')
        
        # --- NÂNG CẤP: ĐƯỜNG DẪN LƯU MODEL ---
        model_save_path = f"ML251/models/{model_name_safe}_finetuned"
        os.makedirs("ML251/models", exist_ok=True)

        y_train_mapped, y_test_mapped = y_train - 1, y_test - 1
        num_labels = len(y_train.unique())
        tokenizer = AutoTokenizer.from_pretrained(model_name) # Tokenizer chung
        train_dataset = ReviewDataset(tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=256), y_train_mapped.tolist())
        test_dataset = ReviewDataset(tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=256), y_test_mapped.tolist())

        # --- NÂNG CẤP: CHUYỂN SANG TENSORBOARD LOGGING ---
        training_args = TrainingArguments(
            output_dir='./results/checkpoints', num_train_epochs=2,
            per_device_train_batch_size=16, per_device_eval_batch_size=64,
            warmup_steps=500, weight_decay=0.01,
            logging_dir=f'./tensorboard_logs/{model_name_safe}_finetuned', # Log cho TensorBoard
            logging_strategy="steps", logging_steps=50,
            evaluation_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="f1"
        )
        
        start_time = time.time()
        
        # --- NÂNG CẤP: LOGIC SAVE/LOAD CHO FINE-TUNED MODEL ---
        if os.path.exists(model_save_path):
            print(f"  > Tải mô hình fine-tuned đã có từ: {model_save_path}")
            model = AutoModelForSequenceClassification.from_pretrained(model_save_path).to(self.device)
            trainer = Trainer(model=model, args=training_args, eval_dataset=test_dataset, compute_metrics=compute_metrics)
            print("  > Chỉ chạy đánh giá (evaluate) trên mô hình đã tải.")
            eval_results = trainer.evaluate()
        else:
            print(f"  > Không tìm thấy mô hình đã lưu. Bắt đầu huấn luyện từ đầu...")
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(self.device)
            trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=compute_metrics)
            trainer.train()
            print(f"  > Huấn luyện hoàn tất. Đang lưu model vào: {model_save_path}")
            trainer.save_model(model_save_path)
            eval_results = trainer.evaluate()
        # --- KẾT THÚC NÂNG CẤP ---

        duration = time.time() - start_time
        
        self.results.append({
            "pipeline_type": "deep_learning_finetune", "embedding_model": model_name,
            "classifier": "fine-tuned_head", "accuracy": eval_results['eval_accuracy'],
            "f1_score": eval_results['eval_f1'], "training_time_seconds": duration
        })
        print(f"    -> Hoàn thành trong {duration:.2f} giây. F1-Score: {eval_results['eval_f1']:.4f}")

    def get_results(self): return pd.DataFrame(self.results)
    
# ==============================================================================
# CLASS 2: SequenceModelRunner (MỚI - Dành cho Word2Vec/GloVe + RNN/LSTM)
# ==============================================================================

# --- PyTorch Model Definition ---
class SequenceClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, model_type='lstm', n_layers=2, dropout=0.5):
        super().__init__()
        RecurrentLayer = nn.LSTM if model_type == 'lstm' else nn.RNN
        
        self.recurrent = RecurrentLayer(embedding_dim, hidden_dim, num_layers=n_layers,
                                        bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text_vectors):
        # text_vectors = [batch_size, seq_len, embedding_dim]
        packed_output, (hidden, cell) = self.recurrent(text_vectors)
        
        # Concat the final forward and backward hidden states
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        # hidden = [batch_size, hidden_dim * 2]
            
        return self.fc(hidden)

class SequenceModelRunner:
    def __init__(self, config):
        self.config = config
        self.results = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Sequence Model Runner sử dụng thiết bị: {self.device}")

    def _load_glove_embeddings(self, path):
        print(f"  > Đang tải GloVe embeddings từ {path}...")
        embeddings_dict = {}
        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
        return embeddings_dict

    def _texts_to_sequences(self, texts, embedding_model, max_len=128):
        print(f"  > Chuyển đổi văn bản thành chuỗi vector (max_len={max_len})...")
        embedding_dim = len(next(iter(embedding_model.values())))
        sequences = []
        
        for text in tqdm(texts, desc="Processing texts"):
            tokens = nltk.word_tokenize(text.lower())
            vectors = [embedding_model.get(token, np.zeros(embedding_dim)) for token in tokens[:max_len]]
            
            # Padding
            if len(vectors) < max_len:
                vectors.extend([np.zeros(embedding_dim)] * (max_len - len(vectors)))
            
            sequences.append(vectors)
            
        return np.array(sequences, dtype=np.float32)

    def run_experiment(self, X_train, y_train, X_test, y_test):
        embedding_configs = self.config.get("embedding_models", [])
        model_types = self.config.get("models", [])
        
        for embed_config in embedding_configs:
            embed_type = embed_config['type']
            embed_path = embed_config['path']
            
            # --- Caching logic cho sequences ---
            sequence_file = f"ML251/features/{embed_type}_sequences.npz"
            if os.path.exists(sequence_file):
                print(f"Tải sequences đã xử lý từ cache: {sequence_file}")
                data = np.load(sequence_file)
                X_train_seq, X_test_seq = data['train'], data['test']
            else:
                print(f"Bắt đầu xử lý cho embedding type: {embed_type}")
                if embed_type == 'glove':
                    embedding_model = self._load_glove_embeddings(embed_path)
                elif embed_type == 'word2vec':
                    # Gensim's Word2Vec loader would go here
                    raise NotImplementedError("Word2Vec loader not implemented yet. Please use GloVe.")
                else:
                    continue
                    
                X_train_seq = self._texts_to_sequences(X_train, embedding_model)
                X_test_seq = self._texts_to_sequences(X_test, embedding_model)
                np.savez(sequence_file, train=X_train_seq, test=X_test_seq)
                print(f"Đã lưu sequences vào cache: {sequence_file}")

            embedding_dim = X_train_seq.shape[2]
            num_labels = len(y_train.unique())
            y_train_mapped = torch.tensor(y_train.values - 1, dtype=torch.long)
            y_test_mapped = torch.tensor(y_test.values - 1, dtype=torch.long)
            
            train_data = TensorDataset(torch.from_numpy(X_train_seq), y_train_mapped)
            test_data = TensorDataset(torch.from_numpy(X_test_seq), y_test_mapped)
            
            train_loader = DataLoader(train_data, shuffle=True, batch_size=64)
            test_loader = DataLoader(test_data, batch_size=128)
            
            for model_type in model_types:
                print(f"\n--- Bắt đầu thử nghiệm: {embed_type} + {model_type} ---")
                start_time = time.time()
                
                model = SequenceClassifier(embedding_dim, hidden_dim=128, output_dim=num_labels, model_type=model_type).to(self.device)
                optimizer = torch.optim.Adam(model.parameters())
                criterion = nn.CrossEntropyLoss().to(self.device)
                
                # Training Loop
                model.train()
                for epoch in range(3): # Train for 3 epochs
                    for texts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                        texts, labels = texts.to(self.device), labels.to(self.device)
                        optimizer.zero_grad()
                        output = model(texts)
                        loss = criterion(output, labels)
                        loss.backward()
                        optimizer.step()
                
                # Evaluation Loop
                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for texts, labels in test_loader:
                        texts, labels = texts.to(self.device), labels.to(self.device)
                        output = model(texts)
                        preds = output.argmax(dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                end_time = time.time()
                duration = end_time - start_time
                
                f1 = f1_score(all_labels, all_preds, average='weighted')
                accuracy = accuracy_score(all_labels, all_preds)

                self.results.append({
                    "pipeline_type": "sequence_model",
                    "embedding_model": embed_type,
                    "model": model_type,
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "training_time_seconds": duration
                })
                print(f"  -> Hoàn thành trong {duration:.2f} giây. F1-Score: {f1:.4f}")

    def get_results(self):
        return pd.DataFrame(self.results)