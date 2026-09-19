from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def load_artifact(path: str | Path):
    return joblib.load(path)


def predict_one(artifact, row: dict):
    X = pd.DataFrame([row])
    prob = float(artifact["model"].predict_proba(X)[:, 1][0])
    threshold = float(artifact.get("threshold", 0.50))
    return {
        "probability_gt_50k": prob,
        "threshold": threshold,
        "predicted_income_status": ">50K" if prob >= threshold else "<=50K",
    }


def main():
    p = argparse.ArgumentParser(description="Predict Adult income status from a JSON object/file.")
    p.add_argument("--model", default="models/best_model.joblib")
    p.add_argument("--json", required=True, help="Path to a JSON file containing one observation.")
    args = p.parse_args()

    artifact = load_artifact(args.model)
    row = json.loads(Path(args.json).read_text(encoding="utf-8"))
    print(json.dumps(predict_one(artifact, row), indent=2))


if __name__ == "__main__":
    main()
