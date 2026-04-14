from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "lung_cancer_samples.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "models"


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load and split features/target."""
    df = pd.read_csv(csv_path)
    if "diagnosis" not in df.columns:
        raise ValueError("Expected a 'diagnosis' target column in the dataset.")
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    return X, y


def build_pipeline(feature_names: list[str]) -> Pipeline:
    """Create a preprocessing + classification pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_names),
        ]
    )
    model = LogisticRegression(max_iter=500, class_weight="balanced")
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def train(csv_path: Path, test_size: float, random_state: int) -> dict[str, float]:
    """Train the model, persist artifacts, and return metrics."""
    X, y = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_pipeline(feature_names=list(X.columns))
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACTS_DIR / "lung_cancer_pipeline.joblib")

    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lung cancer detection model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of the dataset used for validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic splits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train(args.data, args.test_size, args.random_state)
    print("Training complete. ROC-AUC: {:.3f}".format(metrics["roc_auc"]))


if __name__ == "__main__":
    main()

