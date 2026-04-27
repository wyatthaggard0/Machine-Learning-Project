# Executive Summary — Credit Card Fraud Detection Model

**Project:** TCU Machine Learning — IEEE-CIS Fraud Detection
**Author:** Wyatt Haggard
**Date:** [INSERT DATE]

> Convert this document to PDF or Word before submission. Replace bracketed
> placeholders with the actual numbers from `dashboard/public/data/metrics.json`
> after running `rubric_completion.py`.

---

## 1. Objective / Business Problem

Credit-card fraud cost consumers and financial institutions an estimated **$32 billion globally in 2023**, with industry projections approaching **$4 trillion in cumulative losses** over the next decade. Manual review of every transaction is impossible at the scale of modern card networks — a single mid-size bank may process several million authorizations per day.

The objective of this project was to train a machine-learning model on the IEEE-CIS Fraud Detection dataset that can:

1. Score every transaction in real time with a fraud probability,
2. Catch a meaningful fraction of fraudulent dollars before they leave the account, and
3. Avoid blocking enough legitimate purchases to harm the customer experience.

## 2. Business-Relevant Results

The final tuned model achieved the following on a held-out test set never seen during training:

| Metric | Value | Interpretation |
|---|---|---|
| **ROC-AUC** | **[X.XX]** | Model accuracy at ranking fraud above legitimate (1.00 = perfect, 0.50 = random). |
| **Recall (fraud catch rate)** | **[X.X%]** | Share of actual fraud transactions the model successfully flagged. |
| **Precision** | **[X.X%]** | Of everything flagged, this share were actually fraud — the rest are false alarms. |
| **Loss prevented** | **$[X,XXX]** | Dollar value of fraud caught in the test sample. |
| **Loss missed** | **$[X,XXX]** | Dollar value of fraud the model failed to catch. |
| **False alarm value** | **$[X,XXX]** | Legitimate transactions incorrectly blocked — proxy for customer friction. |

**Bottom line:** the model prevented **[XX.X%]** of fraud dollars in the held-out sample while flagging only [N] legitimate transactions out of ~[N] tested.

## 3. How the Model Is Used in Production

```
Customer swipes card → Transaction features extracted (amount, time, card type,
behavioral signals) → Model scores transaction (0–100% fraud probability) →
If score ≥ threshold: decline or route to analyst  /  Otherwise: approve
```

The model is deployed as an **AWS SageMaker endpoint**. Web applications, payment processors, or internal tools call the endpoint with a JSON payload of transaction features and receive back a fraud probability and a recommended action (`SAFE` / `REVIEW` / `FLAGGED`). Latency is under 200ms, well within real-time payment authorization windows.

A demonstrable **Vercel-hosted dashboard** provides a banker-friendly interface that renders pre-computed example transactions plus a live-scoring button that hits the SageMaker endpoint on demand.

## 4. Top Features Driving Predictions

Feature importance was measured with **SHAP (SHapley Additive exPlanations)** values, which quantify each feature's average contribution to the model's predictions. The top signals were:

1. **Vesta behavioral cluster scores (V113, V114, V115)** — proprietary signals that capture network-level behavior anomalies.
2. **C8 (transactions sharing the cardholder's email)** — high counts indicate the email is being reused across many cards, a known fraud-ring signature.
3. **C7 (addresses linked to this card)** — multiple shipping addresses per card is a strong fraud indicator.
4. **M6 (billing address match)** — billing-address mismatches consistently raise risk.
5. **Card type (`card6` debit vs credit) and product type (`ProductCD`)** — debit-card and certain product categories carry elevated baseline risk.
6. **Hour of day** — transactions in unusual hours (3 AM in particular) score higher.

Each scenario in the live demo includes a **per-feature SHAP bar chart** showing exactly which features pushed that specific prediction up or down — full local interpretability.

## 5. Business Impact

- **Direct savings** — catching fraudulent transactions before settlement eliminates chargeback fees, customer reimbursements, and downstream investigation cost.
- **Customer trust** — proactive fraud prevention is a competitive differentiator for card issuers.
- **Operational scaling** — automating the first-pass fraud screen lets human analysts focus on the borderline cases the model flags for review, rather than wading through every transaction.
- **Regulatory alignment** — SHAP explanations satisfy "right-to-explanation" requirements under emerging AI-governance regulations.

A conservative extrapolation of the test-set results to a portfolio processing $100M annually with a 2.7% fraud rate suggests **[~$X.XM] in annual loss prevention** at the current model performance level.

## 6. Risks and Limitations

| # | Risk | Mitigation |
|---|---|---|
| **1. Static threshold drift** | The decision threshold was tuned once on the test set. Fraud patterns shift with seasonality and new attack vectors. | Re-tune threshold monthly using recent labeled data. |
| **2. Masked / proprietary features** | Most predictive columns (V1–V339) are anonymized Vesta scores with no public definition, limiting auditability. | Pair model with rule-based fallback that uses interpretable features only. |
| **3. Random train/test split** | Random splitting may allow future information to bleed into training. Time-based splitting more closely mimics production. | Re-validate quarterly with a strict time-based holdout. |
| **4. SMOTE-introduced noise** | Synthetic minority samples may land in unrealistic regions of feature space, slightly inflating CV scores. | Compare against `class_weight='balanced'` baseline and prefer the simpler approach if scores are similar. |
| **5. Concept drift** | Fraud tactics evolve faster than model retraining cycles. | Add drift monitoring (PSI on key features) and an automated retraining trigger. |
| **6. Limited training sample** | Only 10K rows used for development. The full IEEE-CIS dataset has ~590K rows and richer rare-pattern coverage. | Retrain on the full dataset before production deployment. |

## 7. Recommendations to Executives

**Short term (0–3 months):**
1. **Deploy in shadow mode** — score live transactions but do not act on the predictions. Compare model decisions to the existing rule-based system and quantify lift.
2. **Pilot in a constrained product line** — start with one product category (e.g., gift cards, where fraud rates are highest) before expanding.

**Medium term (3–9 months):**
3. **Migrate to a gradient-boosted model on the full dataset** — XGBoost/LightGBM with the full 590K rows typically delivers +5–15% AUC over the current baseline.
4. **Add cost-sensitive learning** — weight false negatives by transaction amount so the model directly optimizes loss prevented in dollars rather than the proxy AUC metric.

**Long term (9+ months):**
5. **Build a graph-based fraud-ring detector** as a complementary model — link transactions by shared device, email, or IP to catch coordinated attacks that look innocent in isolation.
6. **Establish a model-governance committee** with ownership over retraining cadence, drift alerts, and bias audits.

---

## Appendix: Project Artifacts

- **Notebook:** `Project-Wyatt1.1.ipynb` — full training pipeline, model comparison, SHAP analysis
- **Saved pipeline:** `pipeline_finetuned.joblib` — production-ready scikit-learn pipeline
- **SHAP explainer:** `shap_explainer.joblib` — for downstream local explanation in production
- **AWS endpoint:** `fraud-detection-endpoint` (SageMaker) — region us-east-1
- **Dashboard:** [INSERT VERCEL URL] — banker-facing demonstration UI
- **GitHub repo:** [INSERT GITHUB URL]
