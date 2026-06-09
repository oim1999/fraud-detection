# Fraud Detection – Adey Innovations Inc.

> **Week 5 & 6 Challenge**  
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


## Setup

```bash
# 1. Clone the repo
git clone https://github.com/oim1999/fraud-detection.git
cd fraud-detection

# 2. Create and activate a virtual environment (recommended)
py -3.11 -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Added raw data files to data/raw/


```

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

**Alaa Ali**  

