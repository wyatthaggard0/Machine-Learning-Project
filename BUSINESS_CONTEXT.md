# Paste this content into the FIRST markdown cell of Project-Wyatt1.1.ipynb
# (replaces / strengthens §4.1 General Analysis of the Business Problem)

---

# Project: Real-Time Credit Card Fraud Detection

## §4.1 — General Analysis of the Business Problem

### The Problem
Credit card fraud cost consumers and financial institutions an estimated **$32 billion globally in 2023**, with industry forecasts placing **cumulative card-fraud losses near $4 trillion over the next decade**. As card networks process millions of transactions per day per institution, manual review of every authorization is impossible. The challenge is to automatically identify the small fraction (~2–4%) of fraudulent transactions in real time, before money leaves the cardholder's account, while keeping false alarms low enough that legitimate customers aren't disrupted.

### Why ML Addresses This Well
Fraud is a textbook **imbalanced binary classification** problem with rich behavioral signals:
- The decision must be made in milliseconds — too fast for a human in the loop.
- The patterns shift constantly as new attack tactics emerge — a model that learns from data adapts faster than hand-coded rules.
- The signal is spread across many weak features (transaction amount, time of day, device, behavioral aggregates), which a model can combine more effectively than a rules engine.

### Approach (Summary)
1. **Data**: IEEE-CIS Fraud Detection dataset (Vesta) — joined transaction + identity tables, ~590K transactions with 434 features.
2. **Pipeline**: cleaning, feature engineering, SMOTE resampling, and model fitting all live inside a single `imblearn.pipeline.Pipeline` to prevent test-set leakage.
3. **Models**: 4 diverse families (Logistic Regression, Random Forest, XGBoost, Naive Bayes) compared on 5-fold stratified CV; best model fine-tuned via GridSearchCV.
4. **Evaluation**: ROC-AUC (threshold-independent and robust to ~3.5% class imbalance), with secondary reporting of Accuracy, Precision, Recall, and F1.
5. **Explainability**: SHAP global summary + local waterfall plots so individual decisions are auditable.
6. **Deployment**: trained pipeline served via AWS SageMaker; banker-facing demonstration UI deployed on Vercel.

### Success Criteria
- **Primary**: Test-set ROC-AUC ≥ 0.92 (industry-acceptable for fraud detection on this dataset).
- **Business**: Loss prevented (dollar value of fraud caught) reported alongside the false-alarm cost so trade-offs are explicit.
- **Operational**: Model + SHAP explainer saved as portable joblib artifacts, deployable as a real-time endpoint.
