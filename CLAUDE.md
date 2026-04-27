# CLAUDE.md

Guidance for Claude Code when working in this repo. Read this fully before editing anything.

---

## 1. Project Goal

This is a **TCU machine learning course project** scoring against a fixed 100-point rubric. The deliverable is a fraud-detection model on the **IEEE-CIS Fraud Detection** dataset, plus a deployed app and an executive summary.

**Primary objective: every rubric line item must be visible and verifiable.** Do not optimize purely for model performance — optimize for *rubric coverage*. A 0.99 AUC means nothing if the rubric checker can't find a SHAP local explainability example or only counts 4 feature-engineering steps in the pipeline.

The full rubric is reproduced in **§4 Rubric Mapping** below — treat it as the source of truth for what counts as "done."

---

## 2. Repo Layout

```
ML-PROJECT/
├── Project-Wyatt1.1.ipynb     ← Technical Report (main deliverable)
├── inference.py               ← SageMaker inference handler (model_fn, predict_fn, etc.)
├── sagemaker_deploy.py        ← AWS deployment script (5 rubric pts)
├── train_identity.csv         ← IEEE-CIS raw data
├── train_transaction.csv      ← IEEE-CIS raw data
├── CLAUDE.md                  ← (this file)
└── dashboard/                 ← Next.js app, deploys to Vercel (replaces Streamlit)
    ├── public/data/
    │   ├── metrics.json       ← model perf table for dashboard
    │   ├── scenarios.json     ← pre-computed example predictions
    │   ├── summary.json       ← executive summary content
    │   └── top_features.json  ← feature importance + SHAP values
    ├── src/                   ← React/Next pages & components
    ├── next.config.js
    ├── package.json
    ├── postcss.config.js
    ├── tailwind.config.js
    └── vercel.json
```

### Streamlit substitution — read this

The rubric says "Streamlit Web App." **The professor has approved a Next.js + Vercel substitute** for this student. Do not "fix" this by suggesting Streamlit. Treat the Next.js dashboard as the rubric's "Web App" and ensure it satisfies both required sub-items: **Prediction (5 pts)** and **SHAP plot (5 pts)**.

### Data flow

Heavy compute happens **only** in the notebook. The notebook exports artifacts:

- Fitted pipeline (`.joblib`) → consumed by `inference.py` on SageMaker
- SHAP explainer (`.joblib`) → consumed by `inference.py` on SageMaker
- `metrics.json`, `scenarios.json`, `summary.json`, `top_features.json` → committed to `dashboard/public/data/` and read by the dashboard at build/render time

Vercel **cannot** run sklearn/XGBoost/SHAP. Do not try to import Python ML libraries in the dashboard. Either (a) read pre-computed JSON, or (b) call the SageMaker endpoint via a Next.js API route.

---

## 3. Tech Stack

- **Notebook**: Python, pandas, scikit-learn, imbalanced-learn (`imblearn.pipeline.Pipeline`), XGBoost, LightGBM, SHAP, matplotlib/seaborn
- **Inference**: AWS SageMaker (sklearn or XGBoost container), `inference.py` handler
- **Dashboard**: Next.js, React, TailwindCSS, deployed to Vercel
- **Pipelines**: must be `imblearn.pipeline.Pipeline` (not `sklearn.pipeline.Pipeline`) so resampling steps live inside the pipeline and respect train/test boundaries

---

## 4. Rubric Mapping (100 pts)

Every line below has a location. If a line has no clear location in the notebook, **flag it** before writing code.

### 4.1 General Analysis of the Business Problem (2.5 pts)
- **Where**: Notebook, top markdown cell
- **Must contain**: what the problem is, why fraud detection matters ($4T global loss figure from the project brief), how ML addresses it, project approach

### 4.2 Data Collection (2.5 pts)
- **Imports (1.25)**: dedicated cell, all libraries imported up front
- **Loading (1.25)**: `train_identity.csv` + `train_transaction.csv` loaded and **merged on `TransactionID`** (left join transaction ← identity)

### 4.3 Data Cleaning — minimum 5 steps × 2 pts = 10 pts
**Must live inside the sklearn pipeline** (rubric explicitly calls this out to prevent leakage). Use `ColumnTransformer` + custom transformers. Suggested 5+:
1. Missing value imputation (numeric: median; categorical: "missing" constant)
2. Outlier handling (winsorization or IQR clipping on `TransactionAmt`)
3. Data type fixes (e.g., card1–6 to category)
4. Categorical encoding (one-hot for low cardinality, target/frequency encoding for high cardinality)
5. Scale transformation (log1p on `TransactionAmt`, StandardScaler on numerics)

### 4.4 Feature Engineering — minimum 10 steps × 2 pts = 20 pts
Also in the pipeline. Mix all three categories:

**Sanitization**
1. Drop business-irrelevant columns
2. Drop columns with >X% missing (e.g., >90%)
3. Drop near-constant features (variance threshold)
4. Drop high-cardinality categoricals beyond a threshold
5. Drop features with low correlation to `isFraud`
6. Adversarial validation drop (train-vs-test classifier; drop top drift features)

**Creative**
7. Decompose `TransactionDT` → hour-of-day, day-of-week, day, week
8. Aggregations: mean/std `TransactionAmt` per `card1`, per `addr1`
9. Frequency encodings (count of each card1, etc.)
10. Interaction features (e.g., `card1_addr1`, amount × productCD)
11. Ratio features (e.g., `TransactionAmt` / mean amount per card)

**Final selection**
12. Correlation/VIF removal of collinear features
13. SelectKBest or Mutual Information
14. RFE on a base model
15. PCA on the V-columns (V1–V339) if dimensionality remains high

Aim for ~12–14 distinct steps so the grader has obvious headroom past the 10 minimum.

### 4.5 Data Visualization — minimum 5 plots × 1 pt = 5 pts
Mix univariate, bivariate, multivariate:
1. **Univariate**: bar chart of `isFraud` (class imbalance)
2. **Univariate**: histogram/KDE of `TransactionAmt` (log-scaled)
3. **Bivariate**: fraud rate by `ProductCD`
4. **Bivariate**: KDE of `TransactionAmt` split by `isFraud`
5. **Bivariate**: correlation heatmap of top features
6. **Multivariate**: scatter / pair plot with `isFraud` as hue

Hit 6+ to be safe.

### 4.6 Models (28.75 pts)

**Resampling step in pipeline (2 pts)** — use `imblearn`'s `SMOTE` or class weights. Must be a pipeline step so it only runs on train folds during CV.

**Train 4 diverse models (4 × 2 = 8 pts)** — pick to satisfy the rubric's "diverse mix" criterion:
- **Logistic Regression** — numerical / parametric
- **Random Forest** — ensemble bagging, robust to overfitting
- **XGBoost or LightGBM** — ensemble boosting
- **Naive Bayes or KNN** — probabilistic or instance-based (for representation diversity)

Each in its own pipeline. Don't just swap the final estimator — keep the full pipeline per model so cleaning + FE + resampling all run inside CV.

- **Scoring metric chosen and justified (1 pt)** — use **ROC AUC** (justify: severe class imbalance, ~3.5% positives, threshold-independent)
- **Train/test split (1 pt)** — stratified, 80/20, fixed `random_state`
- **K-fold CV results (2 pts)** — `StratifiedKFold(n_splits=5)`, plot box plots of AUC across folds for all 4 models
- **Test results across 4 metrics (4 pts)** — Accuracy, Precision, Recall, ROC AUC (and ideally F1) reported in a single comparison DataFrame

**Best model selection (2 pts)** — pick by AUC, **explicitly compare train vs test AUC to diagnose overfitting**, and write a sentence about how it's addressed (regularization / robust algorithm / etc.).

**Fine-tune the best pipeline (8.75 pts)**
- GridSearchCV varying **at least 4 parameters** (5 pts) — e.g., for XGBoost: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
- Save the fine-tuned pipeline with `joblib.dump` (1.25 pts) — filename `pipeline_finetuned.joblib`
- Test-set results across 4 metrics for the tuned model (2.5 pts)

### 4.7 Finalize Model (6.25 pts)
- **Feature ranking (2.5)** — global SHAP summary plot + a top-N table; also save as `top_features.json` for the dashboard
- **Local explainability (2.5)** — SHAP force plot or waterfall for at least one fraud and one non-fraud example. Save these as records in `scenarios.json`
- **Save SHAP Explainer (1.25)** — `joblib.dump(explainer, "shap_explainer.joblib")`

### 4.8 Deployment (15 pts)
- **AWS deployment (5)** — `sagemaker_deploy.py` deploys the model; `inference.py` implements `model_fn`, `input_fn`, `predict_fn`, `output_fn`. Document the endpoint URL/name in the notebook.
- **Web App (10)** — Next.js dashboard on Vercel:
  - **Prediction (5)** — dashboard surfaces predictions. Either pre-computed scenarios from `scenarios.json` rendered in a UI, **or** a form that POSTs to a Next.js API route which proxies to the SageMaker endpoint. Live calls score better demo-wise; pre-computed is safer.
  - **SHAP plot (5)** — dashboard renders a SHAP visualization (bar chart of contributions, or force-plot-style component) per scenario, fed by `top_features.json` and `scenarios.json`.

### 4.9 Conclusion — Executive Summary (10 pts)
Separate document (Word or PDF) covering, in order:
- Objective / business problem
- Business-relevant results metrics (AUC, recall at chosen threshold, expected $ saved)
- How the model is used in production
- Top features driving predictions
- Business impact
- Risks and limitations
- Recommendations to executives

Mirror the same content in `summary.json` so the dashboard exec-summary section stays in sync.

### 4.10 Deliverables checklist
- [ ] Well-organized notebook (Project-Wyatt1.1.ipynb)
- [ ] Executive Summary document
- [ ] Demo of the web app (screenshots or short video — store under `dashboard/public/demo/`)

---

## 5. Conventions & Anti-Patterns

### Always
- Use `imblearn.pipeline.Pipeline`, **never** `sklearn.pipeline.Pipeline` once a resampler is involved.
- All cleaning and feature engineering go **inside** the pipeline. Do not transform the full DataFrame before splitting — that's the leakage failure mode the rubric calls out by name.
- Set `random_state=42` everywhere reproducibility matters (split, CV, models, resampler).
- After every pipeline change, re-run K-fold CV box plots so the notebook narrative stays consistent.
- Keep markdown cells above every code cell explaining *what rubric item this satisfies*. Graders grade fast; make it obvious.

### Never
- Never fit transformers on the full dataset before splitting.
- Never use `.fillna()` / `.drop()` / `.map()` on the raw DataFrame as a "quick clean" — wrap it in a transformer.
- Never import sklearn / xgboost / shap from `dashboard/`. The dashboard is read-only display + optional API proxy.
- Never delete a rubric-required artifact to "clean up." Every `.joblib` and `.json` listed in §4 is required.
- Never replace the Vercel deployment with Streamlit. The professor approved this substitution.

### Naming
- Saved artifacts: `pipeline_finetuned.joblib`, `shap_explainer.joblib`
- JSON exports go in `dashboard/public/data/` (committed) so Vercel builds find them
- Notebook section headers should match rubric section names (e.g., "## 4.6 Models — Train 4 Diverse Pipelines")

---

## 6. How to Approach Common Tasks

**"Add another feature engineering step"**
→ Add it as a transformer step inside the pipeline, not as a DataFrame mutation. Bump the count in the notebook header so the grader sees ≥10.

**"Improve model performance"**
→ First check the rubric's "too high / too low" diagnosis section. If AUC is suspiciously high (>0.99), look for leakage (a feature derived post-label, or fit-on-full-data). If AUC is too low, expand grid search or add interaction features — don't add new model families unless the 4-model diversity requirement is still satisfied.

**"Deploy to SageMaker"**
→ Verify `inference.py` returns probabilities (not just class labels) so SHAP can run downstream. Confirm the artifact bundle includes both `pipeline_finetuned.joblib` and `shap_explainer.joblib`.

**"Update the dashboard"**
→ If new data is needed, regenerate the relevant JSON in the notebook, commit it under `dashboard/public/data/`, then update the React component. Never hardcode metrics in JSX.

**"The grader can't find X"**
→ Add an explicit markdown header in the notebook naming the rubric item, even if the code already does X. Discoverability is part of the grade.

---

## 7. Pre-Submission Checklist

Run through this before each milestone push.

**Notebook**
- [ ] Business problem section present at top (§4.1)
- [ ] All imports in one cell (§4.2)
- [ ] `train_identity` + `train_transaction` merged (§4.2)
- [ ] Pipeline contains ≥5 cleaning steps (§4.3)
- [ ] Pipeline contains ≥10 FE steps (§4.4)
- [ ] ≥5 plots, mixing univariate/bivariate/multivariate (§4.5)
- [ ] Resampling step in pipeline (§4.6)
- [ ] 4 diverse model pipelines trained (§4.6)
- [ ] Scoring metric explicitly named and justified (§4.6)
- [ ] Stratified train/test split with seed (§4.6)
- [ ] K-fold CV box plot rendered (§4.6)
- [ ] Test-set 4-metric table for all 4 models (§4.6)
- [ ] Best model chosen with overfitting commentary (§4.6)
- [ ] GridSearchCV varies ≥4 parameters (§4.6)
- [ ] Tuned pipeline saved as `pipeline_finetuned.joblib` (§4.6)
- [ ] Tuned-model 4-metric table on test set (§4.6)
- [ ] Global feature importance / SHAP summary (§4.7)
- [ ] Local SHAP explanation for ≥1 fraud + ≥1 non-fraud (§4.7)
- [ ] `shap_explainer.joblib` saved (§4.7)

**Deployment**
- [ ] `sagemaker_deploy.py` runs end-to-end and returns an endpoint
- [ ] `inference.py` handlers tested against a sample payload
- [ ] Dashboard deployed to Vercel, public URL recorded
- [ ] Dashboard prediction surface working (scenarios or live)
- [ ] Dashboard SHAP visualization working

**Documents**
- [ ] Executive Summary written, all 7 sub-points covered (§4.9)
- [ ] Demo screenshots or video in `dashboard/public/demo/`
- [ ] `metrics.json`, `scenarios.json`, `summary.json`, `top_features.json` all freshly regenerated and committed

---

## 8. Quick Sanity Check on Project Health

Healthy AUC range for IEEE-CIS with reasonable FE: **0.92–0.96**.
- Below 0.85 → likely missing aggregation features or resampling not actually running in CV folds.
- Above 0.99 → almost certainly leakage; check whether anything was fit on full data before splitting, or whether a label-derived feature snuck in.