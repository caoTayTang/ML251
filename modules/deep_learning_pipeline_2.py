# modules/deep_learning_pipeline_2.py

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import time
import os

# --- Lớp Dataset cho PyTorch ---
class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# --- Hàm tính toán metrics cho Trainer ---
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}

class DeepLearningPipelineRunner:
    def __init__(self, config):
        """
        Khởi tạo runner với config cho pipeline học sâu.
        """
        self.config = config
        self.results = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Sử dụng thiết bị: {self.device}")

    # --- Hướng tiếp cận A: Static Embeddings ---
    def _extract_static_embeddings(self, texts, model_name, batch_size=32):
        print(f"Bắt đầu trích xuất static embeddings từ '{model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting Embeddings"):
            batch = texts[i:i+batch_size].tolist()
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
            # Lấy embedding của token [CLS]
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)
            
        return np.vstack(all_embeddings)

    def run_static_embedding_experiment(self, X_train, y_train, X_test, y_test):
        model_name = self.config.get("embedding_models", [])[0] # Giả sử chỉ có 1 model
        classifier_name = self.config.get("downstream_approaches", {}).get("static_embedding_with_classifier", [])[0]


        train_embedding_path = 'ML251/features/train_embeddings.npy'
        test_embedding_path = 'ML251/features/test_embeddings.npy'
            
        # Xử lý tập train
        if os.path.exists(train_embedding_path):
            print(f"Đã tìm thấy file embedding có sẵn. Đang tải '{train_embedding_path}'...")
            train_embeddings = np.load(train_embedding_path)
        else:
            print("Không tìm thấy file embedding cho tập train. Bắt đầu trích xuất...")
            train_embeddings = self._extract_static_embeddings(X_train, model_name)
            np.save(train_embedding_path, train_embeddings)
            print(f"Đã trích xuất và lưu vào '{train_embedding_path}'.")

        # Xử lý tập test
        if os.path.exists(test_embedding_path):
            print(f"Đã tìm thấy file embedding có sẵn. Đang tải '{test_embedding_path}'...")
            test_embeddings = np.load(test_embedding_path)
        else:
            print("Không tìm thấy file embedding cho tập test. Bắt đầu trích xuất...")
            test_embeddings = self._extract_static_embeddings(X_test, model_name)
            np.save(test_embedding_path, test_embeddings)
            print(f"Đã trích xuất và lưu vào '{test_embedding_path}'.")
        
        start_time = time.time()
        
        if classifier_name == 'logistic_regression':
            classifier = LogisticRegression(max_iter=1000)
        else:
            raise ValueError(f"Classifier '{classifier_name}' không được hỗ trợ.")
            
        classifier.fit(train_embeddings, y_train)
        y_pred = classifier.predict(test_embeddings)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        end_time = time.time()
        duration = end_time - start_time

        self.results.append({
            "pipeline_type": "deep_learning_static",
            "embedding_model": model_name,
            "classifier": classifier_name,
            "accuracy": accuracy,
            "f1_score": f1,
            "training_time_seconds": duration
        })
        print(f"  - Hoàn thành trong {duration:.2f} giây. F1-Score: {f1:.4f}")

    # --- Hướng tiếp cận B: Fine-tuning ---
    def run_finetuning_experiment(self, X_train, y_train, X_test, y_test):
        model_name = self.config.get("embedding_models", [])[0]
        
        print(f"Bắt đầu fine-tuning mô hình '{model_name}'...")
        start_time = time.time()
        
        # Quan trọng: HuggingFace cần label từ 0. Rating của chúng ta là 1-5 -> Chuyển về 0-4
        y_train_mapped = y_train - 1
        y_test_mapped = y_test - 1
        num_labels = len(y_train.unique())

        # 1. Tokenization
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=256)
        test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=256)

        # 2. Tạo Dataset
        train_dataset = ReviewDataset(train_encodings, y_train_mapped.tolist())
        test_dataset = ReviewDataset(test_encodings, y_test_mapped.tolist())
        
        # 3. Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(self.device)

        # 4. Định nghĩa Training Arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=2,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch"
        )

        # 5. Khởi tạo Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        # 6. Huấn luyện và đánh giá
        trainer.train()
        eval_results = trainer.evaluate()

        end_time = time.time()
        duration = end_time - start_time

        self.results.append({
            "pipeline_type": "deep_learning_finetune",
            "embedding_model": model_name,
            "classifier": "fine-tuned_head",
            "accuracy": eval_results['eval_accuracy'],
            "f1_score": eval_results['eval_f1'],
            "training_time_seconds": duration
        })
        print(f"  - Hoàn thành fine-tuning trong {duration:.2f} giây. F1-Score: {eval_results['eval_f1']:.4f}")

    def get_results(self):
        return pd.DataFrame(self.results)