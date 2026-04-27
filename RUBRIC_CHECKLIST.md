# Rubric Coverage Checklist — Final Submission

## What was added

| File | Purpose | Rubric items it covers |
|------|---------|------------------------|
| `rubric_completion.py` | Run as final notebook cell. Trains 4 diverse models, GridSearchCV, SHAP plots, saves artifacts, regenerates dashboard JSONs | §4.6 (28.75 pts), §4.7 (6.25 pts), §4.8 dashboard data |
| `BUSINESS_CONTEXT.md` | Paste into first markdown cell of notebook | §4.1 (2.5 pts) |
| `EXECUTIVE_SUMMARY.md` | Convert to PDF/Word for submission | §4.9 (10 pts) |
| `dashboard/src/app/page.js` (updated) | Now renders SHAP global importance + per-scenario SHAP local contributions + 4-model comparison chart | §4.8 SHAP plot (5 pts) |

## Final Action List

### 1. In JupyterLab
- [ ] Open `Project-Wyatt1.1.ipynb` and replace the first markdown cell with the contents of `BUSINESS_CONTEXT.md`
- [ ] Run all existing cells top to bottom (this populates `imputer`, `scaler_std`, `final_model`, `feature_names`, `best_thresh`, `X_train_bal`, `X_train_std`, `X_test`, `X_test_std`, `y_train_bal`, `y_test`, `df` in memory)
- [ ] Upload `rubric_completion.py` to JupyterLab
- [ ] Add a new cell at the bottom and run: `exec(open("rubric_completion.py").read())`
- [ ] If `xgboost` or `shap` is missing, install: `!pip install xgboost shap` and re-run

### 2. Download artifacts from JupyterLab
- [ ] `dashboard/public/data/metrics.json`
- [ ] `dashboard/public/data/top_features.json`
- [ ] `dashboard/public/data/scenarios.json`
- [ ] `dashboard/public/data/summary.json`
- [ ] `pipeline_finetuned.joblib` (keep — required artifact, listed in rubric)
- [ ] `shap_explainer.joblib` (keep — required artifact)

### 3. Replace local files
- [ ] Drop the four downloaded JSONs into `dashboard/public/data/` (overwrite)
- [ ] Keep `pipeline_finetuned.joblib` and `shap_explainer.joblib` in the project root (gitignored, but graders may inspect)

### 4. Push the dashboard updates
```bash
git add .
git commit -m "Add rubric completion: 4 models, SHAP, dashboard SHAP visualization"
git push
```
Vercel auto-redeploys.

### 5. Redeploy SageMaker endpoint with the tuned pipeline
- [ ] In JupyterLab, modify `sagemaker_deploy.py` to package `pipeline_finetuned.joblib` instead of the old pipeline (or re-run the deploy script as-is — the script already references the variables in memory)
- [ ] Run `exec(open("sagemaker_deploy.py").read())`

### 6. Executive Summary
- [ ] Open `EXECUTIVE_SUMMARY.md`, fill in the [X.XX] placeholders from the new `metrics.json`
- [ ] Insert your Vercel URL and GitHub URL at the bottom
- [ ] Convert to PDF (recommended) — the rubric calls this "Conclusion — Executive Summary"

### 7. Demo evidence
- [ ] Take 2–3 screenshots of the live Vercel dashboard
- [ ] Place them under `dashboard/public/demo/` and commit
- [ ] Optional: short Loom/screen-recording showing the live AWS scoring button

## Rubric coverage after these steps

| Section | Points | Status |
|---------|-------:|--------|
| 4.1 General Analysis of Business Problem | 2.5 | ✅ Covered by BUSINESS_CONTEXT.md |
| 4.2 Data Collection (imports + load + merge) | 2.5 | ✅ Already present |
| 4.3 Data Cleaning (≥5 steps) | 10 | ✅ Already present in notebook |
| 4.4 Feature Engineering (≥10 steps) | 20 | ✅ Already present in notebook |
| 4.5 Visualization (≥5 plots) | 5 | ✅ Already exceeds — 10+ plots |
| 4.6 Models (4 diverse + CV + GridSearch + tuned table + saved pipeline) | 28.75 | ✅ Covered by rubric_completion.py |
| 4.7 SHAP global + local + saved explainer | 6.25 | ✅ Covered by rubric_completion.py |
| 4.8 AWS Deployment + Web App with prediction + SHAP | 15 | ✅ SageMaker endpoint + Vercel dashboard with SHAP visualization |
| 4.9 Executive Summary (7 sub-points) | 10 | ✅ Covered by EXECUTIVE_SUMMARY.md |
| **TOTAL** | **100** | **✅** |

## Pre-submission sanity check
- [ ] Notebook re-runs cleanly top to bottom
- [ ] `pipeline_finetuned.joblib` exists in project root
- [ ] `shap_explainer.joblib` exists in project root
- [ ] Vercel dashboard shows Model Comparison chart with 4 bars
- [ ] Vercel dashboard shows SHAP feature importance chart
- [ ] Clicking a scenario shows its per-feature SHAP contributions
- [ ] Live AWS scoring button works on at least one scenario
- [ ] Executive Summary has all 7 sub-points filled in
- [ ] Demo screenshots committed under `dashboard/public/demo/`
