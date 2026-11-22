from argparse import ArgumentParser
from pathlib import Path

import joblib
import kagglehub
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


MODELS_DIR = Path(__file__).resolve().parent / "models"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
FEATURES = ZERO_AS_MISSING + ["Pregnancies", "DiabetesPedigreeFunction", "Age"]
TARGET = "Outcome"


def train(csv_path: Path | None) -> None:
    if csv_path is None:
        csv_path = Path(kagglehub.dataset_download("saurabh00007/diabetescsv")) / "diabetes.csv"
    data = pd.read_csv(csv_path)
    fit, test = train_test_split(
        data, test_size=0.2, stratify=data[TARGET], random_state=42,
    )

    preprocessing = ColumnTransformer(
        [("zero_as_missing", SimpleImputer(missing_values=0, strategy="median"), ZERO_AS_MISSING)],
        remainder="passthrough",
    )
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("model", GradientBoostingClassifier(
            n_estimators=140, learning_rate=0.05, max_depth=3,
            min_samples_leaf=8, subsample=0.8, random_state=42,
        )),
    ])
    model.fit(fit[FEATURES], fit[TARGET])
    forecast = model.predict(test[FEATURES])
    print(f"Train: {len(fit)} | Test: {len(test)}")
    print(f"Accuracy: {accuracy_score(test[TARGET], forecast):.4f}")
    print(f"F1:       {f1_score(test[TARGET], forecast):.4f}")
    print(f"ROC AUC:  {roc_auc_score(test[TARGET], model.predict_proba(test[FEATURES])[:, 1]):.4f}")
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.joblib", compress=3)


if __name__ == "__main__":
    parser = ArgumentParser(description="Train the gradient boosting model on the Pima diabetes dataset.")
    parser.add_argument("--csv", type=Path, help="Use a local copy of the Kaggle CSV.")
    train(parser.parse_args().csv)
