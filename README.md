# Tesla Stock Price Prediction using Machine Learning in Python

This project predicts **next-day stock price movement of Tesla (TSLA)** using
Machine Learning models in Python. It focuses on **classification** of whether
the stock will go **up or down tomorrow**, based on historical OHLC data and
engineered features.

The project was developed as part of my **MCA Major Project** at **Aurora’s PG College (MCA)** and implemented during my internship at **Sansah Innovations Pvt. Ltd.**

---

## 🚀 Project Overview

- Problem: Predict if **tomorrow's closing price** of Tesla will be **higher**
  than today's closing price.
- Type: **Binary classification** (Up = 1, Down = 0)
- ML Techniques:
  - **StandardScaler** for feature normalization
  - **Logistic Regression** as baseline model
  - **SVC (Polynomial kernel)** for non-linear separation
  - **XGBClassifier (XGBoost)** as advanced ensemble model
- Frontend: **Streamlit** web app for user-friendly predictions

The system takes basic OHLC inputs and quarter-end information and predicts
whether the Tesla stock is likely to go **up** or **down** on the next trading day.

---

## 🧠 Machine Learning Approach

### 1. Features

From each daily record (OHLC data):

- `open-close` = Open − Close  
- `low-high`   = Low − High  
- `is_quarter_end` = 1 if month % 3 == 0 else 0  

**Target variable:**

- `target = 1` if next day Close > today Close  
- `target = 0` otherwise

### 2. Preprocessing

- Remove unnecessary columns (e.g., `Adj Close`)
- Extract `day`, `month`, `year` from `Date`
- Handle missing values / duplicates
- **Standardize** numerical features using `StandardScaler` so that each
  feature has mean 0 and standard deviation 1.

### 3. Models

- **Logistic Regression**
  - Baseline model
  - Interpretable and fast
- **SVC (poly kernel)**
  - Captures non-linear decision boundary
- **XGBClassifier**
  - Gradient boosting ensemble model
  - Handles non-linearity and complex feature interactions
  - Uses regularization to reduce overfitting

### 4. Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix
- Time-series comparison of predicted vs actual movements

In experiments, **XGBClassifier** achieved higher accuracy
(around **70–75%**) compared to baseline Logistic Regression
(around **62–65%**), with better ROC-AUC and F1-score.

---

## 📂 Project Structure

```text
.
├─ app/
│   └─ app.py               # Streamlit UI for interactive predictions
│
├─ src/
│   ├─ train_model.py       # Training pipeline for ML models
│   └─ utils.py             # Helper functions (preprocessing, feature engineering)
│
├─ models/
│   └─ xgb_model.pkl        # Saved XGBoost model
│
├─ data/
│   └─ Tesla.csv            # Historical Tesla stock data
│
├─ notebooks/
│   └─ eda_and_models.ipynb # EDA + experimentation (optional)
│
├─ README.md
├─ requirements.txt
└─ .gitignore
