# 🧠 Thyroid Recurrence Predictor (XGBoost + Flask)

This project is an **end-to-end machine learning web application** that predicts whether a thyroid cancer patient is likely to experience a recurrence of cancer based on multiple medical features. Built with **Flask**, trained using **XGBoost**, and optimized using **modular coding** practices.

---

## 📁 Project Structure

```
thyroid_prediction_project/
│
├── app.py                      # Flask entry point
├── requirements.txt            # Dependencies
├── data/
│   └── Thyroid_new_data.csv    # Dataset used
├── saved_model/
│   └── xgboost_thyroid_model.pkl
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # CSV reader
│   ├── preprocess.py           # Data preprocessing & encoding
│   ├── model.py                # Model training & saving
│   ├── predict.py              # Prediction logic
├── templates/
│   └── index.html              # Bootstrap-based UI
└── README.md                   # This file
```

---

## 🚀 Features

- ✅ User-friendly form-based prediction UI
- ✅ Clean modular architecture (EDA → Preprocessing → Modeling → Deployment)
- ✅ XGBoost model with hyperparameter tuning
- ✅ Evaluation reports (Accuracy, F1-Score, Confusion Matrix, ROC)
- ✅ Ready for deployment on **Render**, **Heroku**, or **any VPS**

---

## 📊 Input Features

| Feature Name                 | Type      | Example Value                                   |
|-----------------------------|-----------|-------------------------------------------------|
| Age                         | Numeric   | 45                                              |
| Gender                      | Categorical | F / M                                        |
| Smoking                     | Categorical | Yes / No                                     |
| Smoking History             | Categorical | Yes / No                                     |
| Radiotherapy History        | Categorical | Yes / No                                     |
| Thyroid Function            | Categorical | Euthyroid / Subclinical Hypothyroidism       |
| Physical Examination        | Categorical | Multinodular goiter                           |
| Adenopathy                  | Categorical | No Lympth Adenopathy                          |
| Types of Thyroid Cancer     | Categorical | Micropapillary / Papillary                    |
| Focality                    | Categorical | Uni-Focal / Multi-Focal                       |
| Risk                        | Categorical | Low / High / Intermediate                     |
| Tumor                       | Categorical | tumor that is 1 cm or smaller                 |
| Lymph Nodes                 | Categorical | no evidence of regional lymph node metastasis |
| Cancer Metastasis           | Categorical | no evidence of distant metastasis            |
| Stage                       | Categorical | First-Stage / IVB                             |
| Treatment Response          | Categorical | Excellent / Indeterminate                     |

---

## 📦 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/thyroid-prediction.git
cd thyroid-prediction
```

### 2. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

### 5. Open in browser

Go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## ✅ Model Evaluation Results

XGBoost | Accuracy: 0.9740 | F1 Score: 0.9524
LogisticRegression | Accuracy: 0.9610 | F1 Score: 0.9302
RandomForest | Accuracy: 0.9740 | F1 Score: 0.9524
SVM | Accuracy: 0.9481 | F1 Score: 0.9091
KNN | Accuracy: 0.9351 | F1 Score: 0.8889

| Metric         | Score     |
|----------------|-----------|
| Accuracy       | 97.4%     |
| F1-Score       | 97.3%     |
| Precision/Recall | 96% / 100% (for No Recurrence) |

> Confusion Matrix:
- TN = 55
- TP = 20
- FN = 2
- FP = 0

---

## 🖼️ Screenshots

### 🎯 UI Home

> Form inputs powered by Bootstrap dropdowns

### 🧪 Prediction Form UI
![Prediction UI](/home/sachingpt/thyroid/reports/ui1.png)
![Prediction UI](/home/sachingpt/thyroid/reports/ui2.png)

### 📊 Confusion Matrix
![Confusion Matrix](/home/sachingpt/thyroid/reports/roc_curve.png)

## 🧪 Future Improvements

- Deploy to Render or Hugging Face Spaces
- Store input/output logs in SQLite or Firebase
- Email alert system on high-risk prediction
- Real-time feature tracking dashboard

---

## 📜 License

MIT License © 2025  
Created by [Sachin Gupta](https://github.com/SachinGupta2012)