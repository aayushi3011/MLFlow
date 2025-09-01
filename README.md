# 🛳 Titanic Survival Prediction - MLOps Pipeline

This project implements an **end-to-end MLOps pipeline** for the Titanic dataset using **PySpark, MLflow, DVC, and FastAPI**.  
It demonstrates data preprocessing, model training with experiment tracking, model deployment via API, drift detection, automated retraining, and resource optimization.

---

## 📌 Features
- ✅ **Data Preprocessing** with PySpark (`preprocess_data.py`)  
- ✅ **Model Training** with Spark MLlib RandomForest + Hyperparameter tuning (`train_model.py`)  
- ✅ **Experiment Tracking** using MLflow (metrics, confusion matrix, model registry)  
- ✅ **Reproducible Pipelines** with DVC (`dvc.yaml`)  
- ✅ **Model Deployment** via FastAPI (`deploy_api.py`)  
- ✅ **API Testing** with multiple passengers (`test_api.py`)  
- ✅ **Drift Detection** using KS-test (`drift_detection.py`)  
- ✅ **Automated Retraining** pipeline (`automated_retraining.py`)  
- ✅ **Resource Optimization** experiments (`resource_optimization.py`)  

---

## 📂 Project Structure
```
.
├── data/
│   ├── raw/                 # Original train/test CSV (tracked with DVC)
│   └── processed/           # Preprocessed datasets
├── models/                  # Saved models
├── reports/
│   ├── figures/             # Confusion matrix
│   ├── retrain_log.txt      # Drift & retraining logs
│   └── resource_optimization_results.csv
├── src/
│   ├── preprocess_data.py
│   ├── train_model.py
│   ├── test_api.py
│   ├── deploy_api.py
│   ├── drift_detection.py
│   ├── automated_retraining.py
│   └── resource_optimization.py
├── dvc.yaml
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1️⃣ Clone repo & create environment
```bash
git clone https://github.com/your-username/titanic-mlops.git
cd titanic-mlops
python -m venv .venv310
.venv310\Scripts\activate   # Windows
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Setup DVC
```bash
dvc init
dvc repro
```

### 4️⃣ Start MLflow UI (in a separate terminal)
```bash
mlflow ui
```
Runs at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 5️⃣ Run FastAPI Deployment (in another terminal)
```bash
python src/deploy_api.py
```
API docs at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧪 Usage

### Test API
```bash
python src/test_api.py
```
Saves results in `reports/api_test_results.csv`.

### Drift Detection
```bash
python src/drift_detection.py
```
- Compares distributions between training & test data.  
- If drift detected → triggers `automated_retraining.py`.  
- Logs saved in `reports/retrain_log.txt`.

### Resource Optimization
```bash
python src/resource_optimization.py
```
- Runs experiments with different Spark & RF settings.  
- Results saved in `reports/resource_optimization_results.csv`.

---

## 📊 Results
- MLflow experiment tracking (params, metrics, artifacts)  
- Confusion matrix saved under `reports/figures/`  
- Drift & retraining logs in `reports/retrain_log.txt`  
- API predictions stored in `reports/api_test_results.csv`  

---

## 🏗 Tools & Stack
- **PySpark** (data preprocessing & training)  
- **MLflow** (experiment tracking, model registry)  
- **DVC** (pipeline orchestration, data versioning)  
- **FastAPI** (model serving)  
- **Scikit-learn / SciPy** (metrics & drift detection)  

---

## 🎥 Demo Video
📌 *(To be attached for submission — run through preprocessing, training, MLflow, API, drift detection, retraining)*

---

## 👩‍💻 Author
- Aayushi Maniar - CH24M503 (M.Tech, Industrial AI)
