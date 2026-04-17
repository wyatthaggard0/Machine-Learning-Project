import joblib
import os
import json
import numpy as np


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "pipeline.joblib"))


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        # Expects {"features": [val1, val2, ...30 values in feature_names order]}
        return np.array(data["features"], dtype=float).reshape(1, -1)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    imputer = model["imputer"]
    scaler = model["scaler"]
    clf = model["model"]
    threshold = model["threshold"]

    X = imputer.transform(input_data)
    X = scaler.transform(X)
    prob = float(clf.predict_proba(X)[0, 1])

    return {
        "fraud_probability": round(prob, 4),
        "is_fraud": prob >= threshold,
        "risk_level": "FLAGGED" if prob >= 0.5 else ("REVIEW" if prob >= 0.25 else "SAFE"),
        "threshold": threshold,
    }


def output_fn(prediction, response_content_type):
    return json.dumps(prediction)
