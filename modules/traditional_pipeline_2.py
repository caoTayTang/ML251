# modules/traditional_pipeline_2.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import time

# Ánh xạ tên trong config với các đối tượng sklearn
VECTORIZERS = {
    "tfidf": TfidfVectorizer(),
    "bow": CountVectorizer()
}

MODELS = {
    "naive_bayes": MultinomialNB(),
    "logistic_regression": LogisticRegression(max_iter=1000),
    "svm": LinearSVC(max_iter=2000)
}

class TraditionalPipelineRunner:
    """
    Class để chạy các thử nghiệm với pipeline học máy truyền thống.
    """
    def __init__(self, config):
        """
        Khởi tạo runner với config cho pipeline truyền thống.
        Ví dụ config:
        {
            "feature_extractors": ["tfidf", "bow"],
            "models": ["naive_bayes", "logistic_regression", "svm"]
        }
        """
        self.config = config
        self.results = []

    def run_experiment(self, X_train, y_train, X_test, y_test):
        """
        Chạy toàn bộ các thử nghiệm dựa trên config đã cung cấp.
        """
        print("Bắt đầu chạy các pipeline truyền thống...")
        
        feature_extractors = self.config.get("feature_extractors", [])
        models_to_run = self.config.get("models", [])

        for extractor_name in feature_extractors:
            if extractor_name not in VECTORIZERS:
                print(f"Warning: Bỏ qua feature extractor không xác định '{extractor_name}'")
                continue
            
            for model_name in models_to_run:
                if model_name not in MODELS:
                    print(f"Warning: Bỏ qua mô hình không xác định '{model_name}'")
                    continue
                
                # Bắt đầu một thử nghiệm
                start_time = time.time()
                
                # 1. Định nghĩa pipeline
                pipeline = Pipeline([
                    ('vectorizer', VECTORIZERS[extractor_name]),
                    ('classifier', MODELS[model_name])
                ])
                
                print(f"  > Đang huấn luyện: {extractor_name} + {model_name}...")
                
                # 2. Huấn luyện pipeline
                pipeline.fit(X_train, y_train)
                
                # 3. Đánh giá
                y_pred = pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                end_time = time.time()
                duration = end_time - start_time
                
                # 4. Lưu kết quả
                self.results.append({
                    "pipeline_type": "traditional",
                    "feature_extractor": extractor_name,
                    "model": model_name,
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "training_time_seconds": duration
                })
                print(f"    - Hoàn thành trong {duration:.2f} giây. F1-Score: {f1:.4f}")
        
        print("Tất cả các pipeline truyền thống đã chạy xong!")
        return pd.DataFrame(self.results)