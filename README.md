# Machine Learning Projects – CO3117 (2025)

**Authors:**  
Lê Chí Đại, Nguyễn Quốc Huy, Phạm Lê Tiến Đạt, Võ Văn Thịnh  
(Nhóm **ML4U** – Đại học Bách Khoa TP.HCM)

**Course:** CO3117 – Bài tập lớn môn Học Máy (2025)  
**Instructor:** TS. Lê Thành Sách  

🌐 **Landing Page & Reports:**  🔗 [Landing Pages](https://caotaytang.github.io/ml251/)  

---

## 🚀 Projects Overview

Repo này chứa **4 dự án lớn** trong khuôn khổ môn học *Học Máy – CO3117*:  

| Project | Domain | Status | Landing Page | Colab |
|---------|--------|--------|------------------|-------|
| **BTL1 – Spaceship Titanic** | Tabular Data (Kaggle Challenge) | ✅ Completed | [Tabular Data](https://caotaytang.github.io/ml251/tabular) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NNmRjgSI6SE14mwKW55yWsWSvvfLDNwm) |
| **BTL2 – Text Processing** | NLP / Text Data | 🔜 Upcoming | _TBD_ | ![Colab Badge](https://img.shields.io/badge/Colab-coming--soon-lightgrey?logo=googlecolab&logoColor=white) |
| **BTL3 – Image Recognition** | Computer Vision | 🔜 Upcoming | _TBD_ | ![Colab Badge](https://img.shields.io/badge/Colab-coming--soon-lightgrey?logo=googlecolab&logoColor=white) |
| **Extension – Advanced Topics** | (Ensemble / Hybrid / Extra domain) | 🔜 Upcoming | _TBD_ | ![Colab Badge](https://img.shields.io/badge/Colab-coming--soon-lightgrey?logo=googlecolab&logoColor=white) |

---

## 📊 Nội dung học thuật

Mỗi project được triển khai như một mini-project độc lập, với các bước chính:  

- Phân tích dữ liệu khám phá (EDA)  
- Xử lý dữ liệu & Feature Engineering  
- Xây dựng pipeline Học máy truyền thống và/hoặc Học sâu  
- Huấn luyện, đánh giá & so sánh mô hình  
- Viết báo cáo & thảo luận kết quả  

---

## 📂 Repo Structure

```
ML251/
│── data/                       # Bộ dữ liệu (raw)
│── docs/                       # Tài liệu, hình minh họa, spec
│── features/                   # Các feature được trích xuất từ data
│── modules/                    # Các modules, utils tự viết được tái sử dụng
│── notebooks/      
│── reports/                    # Báo cáo, kết quả, hình ảnh phân tích
```

---

## ▶️ Usage

Clone repo và cài đặt dependencies:

```bash
git clone https://github.com/caoTayTang/ml251.git
cd ml251

# (Tuỳ chọn) tạo môi trường ảo
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate      # Windows

pip install -r requirements.txt
```