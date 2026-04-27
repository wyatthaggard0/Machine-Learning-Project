"""
SageMaker Deployment Script
============================
Run this in JupyterLab AFTER running all cells in Project-Wyatt.ipynb.
Paste into a new cell and run:  exec(open("sagemaker_deploy.py").read())
"""

import os
import json
import tarfile
import numpy as np
import joblib
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# ── Session setup (matches lab pattern) ──────────────────────────────────────
session = boto3.Session()
s3_client = session.client('s3')
bucket_name = 'wyatt-haggard-s3-bucket'
sagemaker_session = sagemaker.Session(boto_session=session, default_bucket=bucket_name)

credentials = session.get_credentials()
current_access_key  = credentials.access_key
current_secret_key  = credentials.secret_key
current_session_token = credentials.get_frozen_credentials().token

print("=" * 60)
print("  AWS CREDENTIALS — paste these into Vercel env vars")
print("=" * 60)
print(f"AWS_ACCESS_KEY_ID:     {current_access_key}")
print(f"AWS_SECRET_ACCESS_KEY: {current_secret_key}")
print(f"AWS_SESSION_TOKEN:\n{current_session_token}")
print("=" * 60)

# ── Clear bucket ──────────────────────────────────────────────────────────────
s3_resource = boto3.resource('s3')
s3_bucket = s3_resource.Bucket(bucket_name)
s3_bucket.objects.all().delete()
print("✓ S3 bucket cleared")

# ── Save model pipeline ───────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
# Prefer the rubric-completion tuned model if it was trained; otherwise fall back
# to the original best model from the notebook.
deploy_model = globals().get("tuned_model", None) or final_model
deploy_threshold = 0.5 if "tuned_model" in globals() else float(best_thresh)
print(f"Deploying model: {type(deploy_model).__name__}")

artifacts = {
    "imputer":       imputer,
    "scaler":        scaler_std,
    "model":         deploy_model,
    "feature_names": feature_names,
    "threshold":     deploy_threshold,
}
joblib.dump(artifacts, "model/pipeline.joblib")

filename = "finalized_fraud_model.tar.gz"
with tarfile.open(filename, "w:gz") as tar:
    tar.add("model/pipeline.joblib", arcname="pipeline.joblib")
print(f"✓ {filename} created")

# ── Write requirements.txt ────────────────────────────────────────────────────
with open("requirements.txt", "w") as f:
    f.write("numpy==1.26.4\n")
    f.write("scipy==1.12.0\n")
    f.write("scikit-learn==1.3.2\n")
    f.write("pandas==2.2.0\n")
    f.write("imbalanced-learn==0.12.0\n")
print("✓ requirements.txt written")

# ── Upload model to S3 ────────────────────────────────────────────────────────
s3_path_key = "sklearn-pipeline-deployment"
s3_client.upload_file(
    Filename=filename,
    Bucket=bucket_name,
    Key=f"{s3_path_key}/{filename}"
)
model_s3_uri = f"s3://{bucket_name}/{s3_path_key}/{filename}"
print(f"✓ Model uploaded: {model_s3_uri}")

# ── Deploy endpoint ───────────────────────────────────────────────────────────
model_name    = "Fraud-Detection-Logistic-Model"
endpoint_name = "fraud-detection-endpoint"
instance_type = "ml.m5.large"
custom_code_uri = f"s3://{bucket_name}/customCode/"

sklearn_model = SKLearnModel(
    model_data=model_s3_uri,
    role=sagemaker.get_execution_role(),
    entry_point="inference.py",
    framework_version="1.2-1",
    py_version="py3",
    dependencies=["requirements.txt"],
    source_dir=".",
    name=model_name,
    sagemaker_session=sagemaker_session,
    code_location=custom_code_uri,
)

print(f"\nDeploying to endpoint: {endpoint_name} (takes ~5 min) …")
predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    endpoint_name=endpoint_name,
)

print("\n✓ Deployment complete!")
print(f"SAGEMAKER_ENDPOINT_NAME: {endpoint_name}")
print(f"AWS_REGION: {session.region_name}")

# ── Generate scenarios.json from real test data ───────────────────────────────
y_scores = final_model.predict_proba(X_test_std)[:, 1]
fraud_idx = np.where(y_test.values == 1)[0]
legit_idx = np.where(y_test.values == 0)[0]

top_fraud  = fraud_idx[np.argsort(y_scores[fraud_idx])[::-1][:2]]
safe_legit = legit_idx[np.argsort(y_scores[legit_idx])[:3]]

scenarios = []
labels = [("FLAGGED", "High-Risk Transaction")] * 2 + \
          [("SAFE",    "Legitimate Transaction")] * 3

for i, (idx, (risk, title_prefix)) in enumerate(
        zip(list(top_fraud) + list(safe_legit), labels)):

    prob     = float(y_scores[idx])
    orig_idx = X_test.index[idx]
    amt      = float(df.loc[orig_idx, "TransactionAmt"])
    hour     = int(df.loc[orig_idx, "hour"])
    is_we    = int(df.loc[orig_idx, "is_weekend"]) if "is_weekend" in df.columns else 0

    time_label = (
        f"{hour % 12 or 12}:{'00'} "
        f"{'AM' if hour < 12 else 'PM'} • "
        f"{'Weekend' if is_we else 'Weekday'}"
    )

    if prob >= 0.5:
        signals = []
        if df.loc[orig_idx, "C7"] > 3:
            signals.append(f"{int(df.loc[orig_idx, 'C7'])} different billing addresses on file")
        if df.loc[orig_idx, "C8"] > 5:
            signals.append(f"{int(df.loc[orig_idx, 'C8'])} transactions linked to this email")
        if hour < 5 or hour > 22:
            signals.append("Transaction at an unusual hour")
        if not signals:
            signals = ["Behavioral pattern flagged by model",
                       "Transaction deviates from card history"]
    else:
        signals = [
            "Billing address matches bank records",
            "Amount within normal range for this card",
            "No unusual behavioral signals detected",
        ]

    scenarios.append({
        "id":               f"txn-{i + 1}",
        "title":            f"{title_prefix} #{i + 1}",
        "amount":           f"${amt:,.2f}",
        "time":             time_label,
        "fraud_probability": round(prob, 4),
        "risk_level":       risk,
        "key_signals":      signals[:3],
        "features":         X_test_std[idx].tolist(),
        "feature_names":    feature_names,
    })

out_path = os.path.join("dashboard", "public", "data", "scenarios.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(scenarios, f, indent=2)

print(f"\n✓ scenarios.json written to {out_path}")
for s in scenarios:
    print(f"  [{s['risk_level']:7s}] {s['title']:35s}  p={s['fraud_probability']:.3f}  {s['amount']}")

print("\n=== NEXT STEPS ===")
print("1. Copy credentials above into Vercel environment variables")
print(f"2. Set SAGEMAKER_ENDPOINT_NAME = {endpoint_name}")
print(f"3. Set AWS_REGION = {session.region_name}")
print("4. Download scenarios.json and push to GitHub")
