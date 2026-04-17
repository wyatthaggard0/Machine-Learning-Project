"""
SageMaker Deployment Script
============================
Run this in JupyterLab AFTER running all cells in Project-Wyatt.ipynb.
It expects the following variables to already exist in your kernel:
    imputer, scaler_std, final_model, feature_names, best_thresh
    X_test, X_test_std, y_test, y_scores_final, y_pred_opt, df

Usage in JupyterLab terminal:
    python sagemaker_deploy.py

Or paste into a new notebook cell and run.
"""

import os
import json
import tarfile
import numpy as np
import pandas as pd
import joblib
import sagemaker
import boto3
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# ── These must exist from Project-Wyatt.ipynb kernel ──────────────────────────
# If running as a standalone script, uncomment and re-run the notebook first.
# from your_notebook import (imputer, scaler_std, final_model, feature_names,
#                             best_thresh, X_test, X_test_std, y_test, df)

ENDPOINT_NAME = "fraud-detection-endpoint"
SKLEARN_VERSION = "1.2-1"


def save_pipeline():
    """Save imputer + scaler + model as a single joblib artifact."""
    os.makedirs("model", exist_ok=True)
    artifacts = {
        "imputer": imputer,
        "scaler": scaler_std,
        "model": final_model,
        "feature_names": feature_names,
        "threshold": float(best_thresh),
    }
    joblib.dump(artifacts, "model/pipeline.joblib")

    with tarfile.open("model.tar.gz", "w:gz") as tar:
        tar.add("model/pipeline.joblib", arcname="pipeline.joblib")

    print("✓ model.tar.gz created")
    print(f"  Features: {len(feature_names)}")
    print(f"  Threshold: {best_thresh:.3f}")
    return "model.tar.gz"


def deploy_endpoint(model_tar_path):
    """Upload model to S3 and deploy a SageMaker endpoint."""
    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()
    bucket = "wyatt-haggard-s3-bucket"
    prefix = "fraud-detection"

    model_s3_uri = sess.upload_data(model_tar_path, bucket=bucket,
                                    key_prefix=f"{prefix}/model")
    print(f"✓ Model uploaded: {model_s3_uri}")

    sklearn_model = SKLearnModel(
        model_data=model_s3_uri,
        role=role,
        entry_point="inference.py",
        framework_version=SKLEARN_VERSION,
        py_version="py3",
        sagemaker_session=sess,
    )

    predictor = sklearn_model.deploy(
        initial_instance_count=1,
        instance_type="ml.t2.medium",
        endpoint_name=ENDPOINT_NAME,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )

    print(f"✓ Endpoint deployed: {predictor.endpoint_name}")
    print(f"\n  Add this to Vercel environment variables:")
    print(f"  SAGEMAKER_ENDPOINT_NAME={predictor.endpoint_name}")
    print(f"  AWS_REGION={sess.boto_region_name}")
    return predictor


def generate_scenarios():
    """
    Pick 5 representative test transactions (2 fraud, 3 legitimate) and save
    as dashboard/public/data/scenarios.json with real model scores.
    """
    y_scores = final_model.predict_proba(X_test_std)[:, 1]
    fraud_idx = np.where(y_test.values == 1)[0]
    legit_idx = np.where(y_test.values == 0)[0]

    # Top 2 highest-scoring fraud + 3 lowest-scoring legitimate
    top_fraud = fraud_idx[np.argsort(y_scores[fraud_idx])[::-1][:2]]
    safe_legit = legit_idx[np.argsort(y_scores[legit_idx])[:3]]

    scenarios = []
    labels = [("FLAGGED", "High-Risk Fraud Transaction")] * 2 + \
             [("SAFE", "Legitimate Transaction")] * 3

    for i, (idx, (risk, title_prefix)) in enumerate(
            zip(list(top_fraud) + list(safe_legit), labels)):

        prob = float(y_scores[idx])
        orig_idx = X_test.index[idx]
        amt = float(df.loc[orig_idx, "TransactionAmt"])
        hour = int(df.loc[orig_idx, "hour"])
        is_weekend = int(df.loc[orig_idx, "is_weekend"]) if "is_weekend" in df else 0

        time_label = (
            f"{hour % 12 or 12}:{'00' if hour < 12 else '30'} "
            f"{'AM' if hour < 12 else 'PM'} • "
            f"{'Weekend' if is_weekend else 'Weekday'}"
        )

        risk_signals = []
        if prob >= 0.5:
            if df.loc[orig_idx, "C7"] > 3:
                risk_signals.append(f"{int(df.loc[orig_idx, 'C7'])} different billing addresses on file")
            if df.loc[orig_idx, "C8"] > 5:
                risk_signals.append(f"{int(df.loc[orig_idx, 'C8'])} transactions linked to this email")
            if hour < 5 or hour > 22:
                risk_signals.append("Transaction at unusual hour")
            if not risk_signals:
                risk_signals.append("Behavioral pattern flagged by model")
                risk_signals.append("Transaction deviates from card history")
        else:
            risk_signals = [
                "Billing address matches bank records",
                "Amount within normal range for this card",
                "No unusual behavioral signals detected",
            ]

        scenarios.append({
            "id": f"txn-{i + 1}",
            "title": f"{title_prefix} #{i + 1}",
            "amount": f"${amt:,.2f}",
            "time": time_label,
            "fraud_probability": round(prob, 4),
            "risk_level": risk,
            "key_signals": risk_signals[:3],
            # Standardized feature vector — used for live API scoring
            "features": X_test_std[idx].tolist(),
            "feature_names": feature_names,
        })

    out_path = os.path.join("dashboard", "public", "data", "scenarios.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(scenarios, f, indent=2)

    print(f"✓ scenarios.json written to {out_path}")
    for s in scenarios:
        print(f"  [{s['risk_level']:7s}] {s['title']:35s}  p={s['fraud_probability']:.3f}  {s['amount']}")


if __name__ == "__main__":
    print("Step 1: Saving model pipeline …")
    tar_path = save_pipeline()

    print("\nStep 2: Generating demo scenarios …")
    generate_scenarios()

    print("\nStep 3: Deploying to SageMaker …")
    print("  (This takes ~5 minutes)")
    predictor = deploy_endpoint(tar_path)

    print("\n✓ All done. Next steps:")
    print("  1. Copy the SAGEMAKER_ENDPOINT_NAME above to Vercel environment variables")
    print("  2. Also add AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION")
    print("  3. git add dashboard/public/data/scenarios.json && git push")
