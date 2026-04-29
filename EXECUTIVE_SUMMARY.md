# Executive Summary — Credit Card Fraud Detection Model

**Project:** TCU Machine Learning — IEEE-CIS Fraud Detection
**Author:** Wyatt Haggard
**Date:** April 2026

---

## 1. Objective / Business Problem

Credit-card fraud cost consumers and financial institutions an estimated **$32 billion globally in 2023**, with industry projections approaching **$4 trillion in cumulative losses** over the next decade. Manual review of every transaction is impossible at the scale of modern card networks — a single mid-size issuer may process several million authorizations per day.

The objective of this project was to train a machine-learning model on the IEEE-CIS Fraud Detection dataset (Vesta) that can:

1. Score every transaction in real time with a fraud probability,
2. Catch a meaningful fraction of fraudulent dollars before they leave the account, and
3. Surface the reasons behind each decision so that risk teams and customers can act on the model's output.

## 2. Business-Relevant Results

The final tuned model was evaluated on a held-out test set of **2,000 transactions** never seen during training. Results at the chosen operating threshold of **0.436**:

| Metric | Value | Interpretation |
|---|---|---|
| **ROC-AUC** | **0.81** | Model accuracy at ranking fraud above legitimate (1.00 = perfect, 0.50 = random). |
| **Recall (fraud catch rate)** | **73.6%** | Share of actual fraud transactions the model successfully flagged. |
| **Precision** | **7.0%** | Of everything the model flagged, this share were actually fraud. |
| **F1** | **12.9%** | Harmonic mean of precision and recall — sensitive to the precision shortfall. |
| **Balanced Accuracy** | **73.6%** | Average accuracy across fraud and legitimate classes. |
| **MCC** | **0.17** | Matthews correlation — modest signal above random under heavy imbalance. |
| **Loss prevented** | **$6,494.84** | Dollar value of fraud caught in the test sample (39 fraud transactions). |
| **Loss missed** | **$2,595.62** | Dollar value of fraud the model failed to catch (14 transactions). |
| **False alarm value** | **$74,941.55** | Legitimate transactions incorrectly blocked (515 transactions). |
| **Total fraud in sample** | **$9,090.46** | 53 fraudulent transactions, 2.65% of test set. |

**Headline finding:** the model recovers **71.4%** of fraud dollars in the held-out sample. However, it does so by flagging a high volume of legitimate transactions — at this threshold the false-alarm cost ($74,941) substantially exceeds the loss prevented ($6,495). The model's high recall and modest AUC make it a solid first-pass screen, but the operating threshold needs to be revisited before any production deployment.

**Generalization check:** train AUC is **0.852** vs test AUC of **0.810**, a gap of **4.2 percentage points**. This is within the healthy range for a regularized linear model with SMOTE balancing — the model is learning real signal rather than memorizing the training set, and performance on truly new transactions should track close to the test-set numbers above.

## 3. How the Model Is Used in Production

```
Customer initiates transaction → Features extracted (amount, card metadata,
behavioral signals) → Model scores 0–100% fraud probability → Score ≥ 0.436:
flagged for analyst review or decline / Below: approve.
```

The model is deployed as an **AWS SageMaker endpoint**. Web applications, payment processors, or internal review tools call the endpoint with a JSON payload of transaction features and receive back a fraud probability and a recommended action (`SAFE` / `REVIEW` / `FLAGGED`). End-to-end latency is sub-second, well within real-time payment authorization windows.

A banker-facing **Vercel-hosted dashboard** demonstrates the model in two modes: pre-computed example transactions with their SHAP explanations, and a "Build Your Own" form where the user composes a transaction and receives a live score from the SageMaker endpoint.

## 4. Top Features Driving Predictions

Feature importance was measured with **SHAP (SHapley Additive exPlanations)** values, which quantify each feature's average contribution to the model's predictions. The signals carrying the most weight in the final model:

1. **Vesta behavioral cluster scores (V113, V114, V115)** — proprietary signals that capture network-level behavior anomalies.
2. **C8 (transactions sharing the cardholder's email)** — high counts indicate the email is being reused across many cards, a known fraud-ring signature.
3. **C7 (addresses linked to this card)** — multiple shipping addresses per card is a strong fraud indicator.
4. **M6 (billing address match)** — billing-address mismatches consistently raise risk.
5. **Card type and product type (`card6`, `ProductCD`)** — debit-card and certain product categories carry elevated baseline risk.

Each prediction in the live dashboard ships with a **per-feature SHAP bar chart** showing which features pushed the specific score up or down — full local interpretability in support of analyst review and customer disputes.

## 5. Business Impact

- **Direct savings.** In the test sample, the model recovered **$6,494** of fraud — 71% of the dollar value at risk. Catching fraud before settlement eliminates chargeback fees, customer reimbursements, and downstream investigation cost.
- **Customer trust.** Proactive fraud prevention is a competitive differentiator for card issuers. Even an imperfect model materially reduces the surface area of successful fraud.
- **Operational scaling.** Automating the first-pass fraud screen lets human analysts focus on the borderline cases the model flags for review, rather than wading through every transaction.
- **Regulatory alignment.** SHAP explanations satisfy "right-to-explanation" requirements under emerging AI-governance regulations; every flagged transaction comes with an auditable per-feature contribution breakdown.

The current threshold is calibrated for high recall, which is appropriate for a fraud-detection prototype but generates a high volume of false alarms. Before production deployment, the threshold should be raised (or a cost-sensitive learning approach adopted) so that the dollar value of false alarms is contained relative to the dollar value of fraud caught.

## 6. Risks and Limitations

| # | Risk | Impact |
|---|---|---|
| **1. Low precision at current threshold** | Of every 100 transactions flagged, only ~7 are actually fraud. The model blocked **$74,941** of legitimate transaction value to save **$6,495** of fraud — a ~12× imbalance. | Customer friction, manual-review backlog, lost revenue. The threshold should be raised to ~0.6–0.7 to bring the false-alarm cost in line with fraud caught. |
| **2. Modest AUC** | Test ROC-AUC of 0.81 is below the typical IEEE-CIS benchmark (0.92–0.96) achieved with gradient-boosted trees on the full dataset. | Indicates room to improve signal extraction. |
| **3. Static threshold drift** | Threshold was tuned once; fraud patterns shift seasonally and with new attack types. | Performance degrades over time without periodic re-calibration. |
| **4. Masked / proprietary features** | Most predictive columns (V1–V339) are anonymized Vesta scores with no public definition. | Reduces auditability; harder to validate fairness. |
| **5. Random train/test split** | The split was random rather than time-based. In production, a model trained on January data must generalize to March fraud patterns. | Optimistic AUC estimates; real-world performance likely lower. |
| **6. Limited training sample** | Only 10K rows used for development; the full IEEE-CIS dataset has ~590K rows with richer rare-pattern coverage. | Higher false-negative rate on novel attack vectors. |

## 7. Recommendations to Executives

**Short term (0–3 months):**
1. **Raise the decision threshold from 0.436 to ~0.65** before any production deployment. This will reduce recall modestly but should cut false-alarm value by an order of magnitude — a much more defensible operating point.
2. **Deploy in shadow mode.** Score live transactions but do not act on the predictions. Compare the model's decisions to the existing rule-based system and quantify lift before turning on enforcement.

**Medium term (3–9 months):**
3. **Retrain on the full 590K-row dataset using a gradient-boosted tree** (XGBoost or LightGBM). Industry experience suggests +5–15 percentage points of AUC and meaningful precision gains over the linear baseline.
4. **Adopt cost-sensitive learning** — weight false negatives by transaction amount so the model directly optimizes loss prevented in dollars rather than the proxy AUC metric.
5. **Move to time-based train/test splits** to produce a more realistic generalization estimate and surface concept drift earlier.

**Long term (9+ months):**
6. **Build a graph-based fraud-ring detector** as a complementary model — link transactions by shared device, email, or IP to catch coordinated attacks that look innocent in isolation.
7. **Establish a model-governance committee** with ownership over retraining cadence, drift monitoring, and bias audits.

---

## Appendix: Project Artifacts

- **Notebook:** `Project-Wyatt1.1.ipynb` — full training pipeline, model comparison, SHAP analysis
- **Saved pipeline:** `pipeline_finetuned.joblib` — production-ready scikit-learn pipeline
- **SHAP explainer:** `shap_explainer.joblib` — for downstream local explanation in production
- **AWS endpoint:** `fraud-detection-endpoint` (SageMaker, region us-east-1)
- **Dashboard:** Vercel-hosted banker demonstration UI
- **GitHub repo:** github.com/wyatthaggard0/Machine-Learning-Project
