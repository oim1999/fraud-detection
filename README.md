# Fraud Detection – Adey Innovations Inc.

> **KAIM Week 5 & 6 Challenge**  
> Improving fraud detection for e-commerce and bank credit card transactions.

---

## Project Overview

This project builds a unified fraud detection system handling two different transaction types:

| Dataset | Rows | Features | Fraud Rate |
|---------|------|----------|------------|
| `Fraud_Data.csv` (e-commerce) | 151,112 | 11 | 9.36% |
| `creditcard.csv` (bank credit) | 284,807 | 31 | 0.17% |

Both datasets are **highly imbalanced** — the minority (fraud) class is a small fraction of total records. This shapes every modeling and evaluation decision in the project.

---

## Repository Structure

```
fraud-detection/
├── data/               ← add to .gitignore — never commit raw data
│   ├── raw/            ← place original CSV files here
│   └── processed/      ← cleaned & engineered data saved here
├── notebooks/          ← one notebook per task stage
│   ├── eda_fraud_data.py         # Task 1 – EDA on e-commerce data
│   ├── eda_creditcard.py         # Task 1 – EDA on credit card data
│   └── feature_engineering.py   # Task 1 – Features, scaling, SMOTE
├── src/
│   └── preprocessing.py         # Reusable helper functions
├── models/             ← saved model artefacts (.pkl files)
├── scripts/            ← standalone utility scripts
├── tests/              ← unit tests
├── requirements.txt
└── README.md
```

> **Note on notebooks:** The `.py` files in `notebooks/` use `# %%` cell markers,  
> which VS Code's Jupyter extension recognises as notebook cells.  
> Open them in VS Code → right-click → *Open as Jupyter Notebook*.  
> Or convert with: `pip install jupytext && jupytext --to notebook notebooks/eda_fraud_data.py`

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/fraud-detection.git
cd fraud-detection

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add raw data files to data/raw/
#    - Fraud_Data.csv
#    - IpAddress_to_Country.csv
#    - creditcard.csv

# 5. Open notebooks in VS Code or Jupyter
jupyter notebook
```

---

## Task Progress

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Data Analysis & Preprocessing | ✅ Complete |
| Task 2 | Model Building & Training | 🔄 In Progress |
| Task 3 | Model Explainability (SHAP) | ⏳ Pending |

---

## Key Findings (Task 1 Summary)

- **E-commerce data:** No missing values or duplicates. Fraud rate is 9.36% — moderate imbalance.  
  Engineered features include `time_since_signup_hours`, `hour_of_day`, `day_of_week`,  
  `user_transaction_count`, and `device_transaction_count`.

- **Credit card data:** 1,081 duplicate rows removed. Fraud rate is 0.17% — severe imbalance.  
  Only `Amount` and `Time` required scaling; V1–V28 are pre-scaled via PCA.

- **Geolocation:** IP addresses converted to integers and merged with country lookup via  
  range-based join (`pd.merge_asof`). Country added as a fraud signal feature.

- **Class imbalance strategy:** SMOTE applied to training splits only.  
  Never applied to test or validation data to prevent data leakage.

---

## Evaluation Metrics

| Metric | Why we use it |
|--------|---------------|
| **AUC-PR** | Area under Precision-Recall curve — best for severe imbalance |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | Shows counts of TP, TN, FP, FN directly |
| ❌ Accuracy | **Not used** — misleading on imbalanced data |

---

## Author

**[Your Name]**  
KAIM – AI Mastery Program, Cohort [X]
