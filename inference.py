import joblib
import os
import json
import numpy as np


def _flatten_shap_row(sv):
    """Reduce a SHAP output to a 1-D class-1 (fraud) feature vector.

    Handles every shape SHAP has emitted across versions for binary models:
      - list of [class_0_arr, class_1_arr]  (older SHAP)
      - ndarray (n_samples, n_features, n_classes)  (newer SHAP, 3-D)
      - ndarray (n_samples, n_features)  (already class-1)
      - ndarray (n_features,)  (already 1-D)
    """
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[0, :, -1]
    if sv.ndim == 2:
        return sv[0]
    return sv


def _flatten_base(base):
    if isinstance(base, (list, np.ndarray)):
        arr = np.atleast_1d(np.asarray(base, dtype=float))
        return float(arr[-1] if arr.size > 1 else arr[0])
    return float(base)


def model_fn(model_dir):
    artifacts = joblib.load(os.path.join(model_dir, "pipeline.joblib"))
    explainer_path = os.path.join(model_dir, "shap_explainer.joblib")
    if os.path.exists(explainer_path):
        try:
            data = joblib.load(explainer_path)
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
        "is_fraud":          prob >= threshold,
        "risk_level":        "FLAGGED" if prob >= 0.5 else ("REVIEW" if prob >= 0.25 else "SAFE"),
        "threshold":         threshold,
    }

    explainer = model.get("explainer")
    if explainer is not None:
        try:
            # KernelExplainer needs nsamples; TreeExplainer ignores it
            try:
                sv_row = _flatten_shap_row(
                    explainer.shap_values(input_data, nsamples=64)
                )
            except TypeError:
                sv_row = _flatten_shap_row(explainer.shap_values(input_data))

            base = _flatten_base(explainer.expected_value)

            order = np.argsort(np.abs(sv_row))[::-1][:6]
            contributions = []
            for i in order:
                idx = int(i)
                fname = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
                contributions.append({
                    "feature":    fname,
                    "shap_value": round(float(sv_row[idx]), 4),
                })
            response["shap_contributions"] = contributions
            response["shap_base_value"]    = round(base, 4)
        except Exception as e:
            print(f"SHAP computation failed: {type(e).__name__}: {e}")
            response["shap_error"] = f"{type(e).__name__}: {e}"

    return response


def output_fn(prediction, response_content_type):
    return json.dumps(prediction)
