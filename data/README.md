# Dataset Information: Credit Card Fraud Detection

This directory contains data specifications and utilities for the **Credit Card Fraud Detection System**.

---

## 1. Official Dataset Details

- **Dataset Name**: Credit Card Fraud Detection
- **Source**: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Origin**: Machine Learning Group (MLG) at ULB (Université Libre de Bruxelles)
- **Observations**: 284,807 credit card transactions made by European cardholders in September 2013
- **Imbalance**: 492 fraud cases (~0.172% of total transactions)

### Schema & Features:
| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `Time` | Float / Int | Number of seconds elapsed between this transaction and the first transaction in the dataset |
| `V1` – `V28` | Float | 28 numerical features obtained via Principal Component Analysis (PCA) for confidentiality |
| `Amount` | Float | Transaction amount in USD/EUR |
| `Class` | Integer (0 or 1) | Target label: `0 = Legitimate transaction`, `1 = Fraudulent transaction` |

---

## 2. Setting Up the Dataset

### Option A: Using the Official Kaggle Dataset
1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Place `creditcard.csv` directly in this `data/` directory:
   ```
   data/creditcard.csv
   ```

### Option B: Automatic Generation (Demo / Benchmark Benchmark Mode)
If you do not have Kaggle credentials downloaded yet, run our built-in generator:
```bash
python data/download_or_generate_data.py
```
This utility generates a representative dataset matching the exact schema (`Time`, `V1`–`V28`, `Amount`, `Class`), preserving the real-world fraud ratio (~0.17%), realistic PCA distributions, and outlier characteristics to allow end-to-end pipeline verification anywhere without internet access.

---

## 3. Data Privacy & Realistic Simulation Notes
- No real personally identifiable information (PII), card numbers, CVVs, or cardholder names are ever stored or processed.
- All features `V1`–`V28` are anonymized numerical projections.
