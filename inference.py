import joblib
import os
import json
import numpy as np


def model_fn(model_dir):
    artifacts = joblib.load(os.path.join(model_dir, "pipeline.joblib"))
    explainer_path = os.path.join(model_dir, "shap_explainer.joblib")
    if os.path.exists(explainer_path):
        try:
            data = joblib.load(explainer_path)
            # If saved as metadata wrapper, rebuild the explainer from the model
            if isinstance(data, dict) and data.get("is_metadata_wrapper"):
                try:
                    import shap
                    artifacts["explainer"] = shap.TreeExplainer(artifacts["model"])
                except Exception as e:
                    print(f"Could not rebuild TreeExplainer: {e}")
            else:
                artifacts["explainer"] = data
        except Exception as e:
            print(f"Could not load shap_explainer: {e}")
    return artifacts


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data["features"], dtype=float).reshape(1, -1)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    clf = model["model"]
    threshold = model["threshold"]
    feature_names = model.get("feature_names", [])

    prob = float(clf.predict_proba(input_data)[0, 1])

    response = {
        "fraud_probability": round(prob, 4),
        "is_fraud": prob >= threshold,
        "risk_level": "FLAGGED" if prob >= 0.5 else ("REVIEW" if prob >= 0.25 else "SAFE"),
        "threshold": threshold,
    }

    # Compute live SHAP values if explainer is available
    explainer = model.get("explainer")
    if explainer is not None:
        try:
            sv = explainer.shap_values(input_data)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            sv_row = sv[0] if (hasattr(sv, "ndim") and sv.ndim == 2) else sv

            base = explainer.expected_value
            if isinstance(base, (list, np.ndarray)):
                base = float(base[1]) if len(np.atleast_1d(base)) > 1 else float(base)

            order = np.argsort(np.abs(sv_row))[::-1][:6]
            contributions = []
            for i in order:
                fname = feature_names[i] if i < len(feature_names) else f"f{i}"
                contributions.append({
                    "feature":    fname,
                    "shap_value": round(float(sv_row[i]), 4),
                })
            response["shap_contributions"] = contributions
            response["shap_base_value"]    = round(float(base), 4)
        except Exception as e:
            response["shap_error"] = str(e)

    return response


def output_fn(prediction, response_content_type):
    return json.dumps(prediction)
