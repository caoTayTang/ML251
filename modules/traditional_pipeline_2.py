import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import time
import os
import numpy as np
from scipy.sparse import save_npz, load_npz

# Ánh xạ tên trong config với các đối tượng sklearn
# Khởi tạo mà không có ngram_range, chúng ta sẽ thiết lập nó sau
VECTORIZERS = {
    "tfidf": TfidfVectorizer(),
    "bow": CountVectorizer()
}

MODELS = {
    "naive_bayes": MultinomialNB(),
    "logistic_regression": LogisticRegression(max_iter=1000),
    "svm": LinearSVC(max_iter=2000, dual=True) # dual=True có thể cần thiết cho một số phiên bản sklearn
}

class TraditionalPipelineRunner:
    """
    Class để chạy các thử nghiệm với pipeline ML truyền thống,
    hỗ trợ N-gram và caching features.
    """
    def __init__(self, config):
        """
        Khởi tạo runner với config cho pipeline truyền thống.
        Config mới hỗ trợ "ngram_ranges".
        """
        self.config = config
        self.results = []
        # Lấy danh sách ngram_ranges từ config, mặc định là (1, 1) (chỉ unigrams)
        self.ngram_ranges = self.config.get("ngram_ranges", [(1, 1)])

    def run_experiment(self, X_train, y_train, X_test, y_test):
        """
        Chạy toàn bộ các thử nghiệm dựa trên config.
        Thứ tự lặp: Extractor -> N-gram -> Model.
        """
        print("Bắt đầu chạy các pipeline truyền thống (với caching)...")
        
        feature_extractors = self.config.get("feature_extractors", [])
        models_to_run = self.config.get("models", [])

        # Tạo thư mục 'features' nếu chưa có để lưu cache
        os.makedirs('features', exist_ok=True)

        for extractor_name in feature_extractors:
            if extractor_name not in VECTORIZERS:
                print(f"Warning: Bỏ qua feature extractor không xác định '{extractor_name}'")
                continue

            for ngram_range in self.ngram_ranges:
                ngram_str = f"{ngram_range[0]}_{ngram_range[1]}"
                
                # --- LOGIC CACHING BẮT ĐẦU ---
                # Tạo đường dẫn file cache duy nhất cho mỗi combination
                train_feature_path = f"features/{extractor_name}_ngram_{ngram_str}_train.npz"
                test_feature_path = f"features/{extractor_name}_ngram_{ngram_str}_test.npz"

                if os.path.exists(train_feature_path) and os.path.exists(test_feature_path):
                    print(f"  > Đang tải features từ cache cho N-gram {ngram_range}...")
                    X_train_vec = load_npz(train_feature_path)
                    X_test_vec = load_npz(test_feature_path)
                else:
                    print(f"  > Trích xuất features: '{extractor_name}' với N-gram {ngram_range}...")
                    
                    # Lấy vectorizer và cấu hình N-gram cho nó
                    vectorizer = VECTORIZERS[extractor_name]
                    vectorizer.set_params(ngram_range=ngram_range)
                    
                    # Fit và transform dữ liệu
                    X_train_vec = vectorizer.fit_transform(X_train)
                    X_test_vec = vectorizer.transform(X_test)
                    
                    # Lưu các ma trận sparse vào file cache
                    print(f"    - Đang lưu features vào cache...")
                    save_npz(train_feature_path, X_train_vec)
                    save_npz(test_feature_path, X_test_vec)

                # --- LOGIC CACHING KẾT THÚC ---

                for model_name in models_to_run:
                    if model_name not in MODELS:
                        print(f"Warning: Bỏ qua mô hình không xác định '{model_name}'")
                        continue
                    
                    start_time = time.time()
                    
                    print(f"    - Đang huấn luyện: {extractor_name} + N-gram {ngram_range} + {model_name}...")
                    
                    # Huấn luyện trực tiếp trên features đã được tính toán/tải
                    model = MODELS[model_name]
                    model.fit(X_train_vec, y_train)
                    
                    # Đánh giá
                    y_pred = model.predict(X_test_vec)
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    # Lưu kết quả, thêm thông tin N-gram
                    self.results.append({
                        "pipeline_type": "traditional",
                        "feature_extractor": extractor_name,
                        "ngram_range": str(ngram_range), # Thêm thông tin ngram
                        "model": model_name,
                        "accuracy": accuracy,
                        "f1_score": f1,
                        "training_time_seconds": duration
                    })
                    print(f"      -> Hoàn thành trong {duration:.2f} giây. F1-Score: {f1:.4f}")
        
        print("\nTất cả các pipeline truyền thống đã chạy xong!")
        return pd.DataFrame(self.results)