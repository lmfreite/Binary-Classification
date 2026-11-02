"""Entrenamiento reproducible de diez modelos para la competencia de churn."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = ROOT / "submissions"


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds stable ratios and non-linear terms available at prediction time."""
    data = frame.copy()
    data["AgeSquared"] = data["Age"] ** 2
    data["CreditScoreAge"] = data["CreditScore"] * data["Age"]
    data["BalanceSalaryRatio"] = data["Balance"] / (data["EstimatedSalary"] + 1.0)
    data["IsZeroBalance"] = (data["Balance"] == 0).astype(int)
    data["ProductsPerAge"] = data["NumOfProducts"] / (data["Age"] + 1.0)
    return data.drop(columns=["id", "CustomerId", "Surname"], errors="ignore")


def make_preprocessor(data: pd.DataFrame) -> ColumnTransformer:
    categorical = data.select_dtypes(include=["object", "string", "category"]).columns
    numeric = data.columns.difference(categorical)
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [
            ("categorical", encoder, categorical.tolist()),
            ("numeric", "passthrough", numeric.tolist()),
        ],
        remainder="drop",
    )


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    def pipeline(model, scale=False):
        steps = [("preprocess", preprocessor)]
        if scale:
            steps.append(("scale", StandardScaler()))
        steps.append(("model", model))
        return Pipeline(steps)

    return {
        "01_logistic_regression": pipeline(
            LogisticRegression(max_iter=1500, C=0.5, class_weight="balanced"),
            scale=True,
        ),
        "02_knn": pipeline(
            KNeighborsClassifier(n_neighbors=75, weights="distance", p=2),
            scale=True,
        ),
        "03_decision_tree": pipeline(
            DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=40,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "04_random_forest": pipeline(
            RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=8,
                max_features=0.8,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
        "05_extra_trees": pipeline(
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_leaf=5,
                max_features=0.9,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
        "06_adaboost": pipeline(
            AdaBoostClassifier(n_estimators=300, learning_rate=0.03, random_state=RANDOM_STATE)
        ),
        "07_hist_gradient_boosting": pipeline(
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=2,
                min_samples_leaf=30,
                random_state=RANDOM_STATE,
            )
        ),
        "08_gradient_boosting": pipeline(
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=20,
                subsample=0.85,
                random_state=RANDOM_STATE,
            )
        ),
        "09_xgboost": pipeline(
            XGBClassifier(
                n_estimators=700,
                max_depth=4,
                learning_rate=0.025,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=8,
                reg_lambda=5,
                reg_alpha=0.1,
                objective="binary:logistic",
                eval_metric="auc",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
        "10_lightgbm": pipeline(
            LGBMClassifier(
                n_estimators=700,
                num_leaves=20,
                max_depth=6,
                learning_rate=0.025,
                min_child_samples=35,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5,
                reg_alpha=0.1,
                verbosity=-1,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
    }


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")

    y = train.pop("Exited").astype(int)
    X = add_features(train)
    X_test = add_features(test)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.22, stratify=y, random_state=RANDOM_STATE
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    records = []
    predictions = {}
    for name, model in build_models(make_preprocessor(X)).items():
        model.fit(X_train, y_train)
        valid_probability = model.predict_proba(X_valid)[:, 1]
        valid_auc = roc_auc_score(y_valid, valid_probability)
        valid_accuracy = accuracy_score(y_valid, valid_probability >= 0.5)
        model.fit(X, y)
        test_probability = model.predict_proba(X_test)[:, 1]
        predictions[name] = test_probability
        submission = sample.copy()
        submission["Exited"] = test_probability
        submission.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        records.append(
            {
                "model": name,
                "validation_auc": valid_auc,
                "validation_accuracy": valid_accuracy,
            }
        )

    scores = pd.DataFrame(records).sort_values("validation_auc").reset_index(drop=True)
    scores.insert(0, "delivery", np.arange(1, len(scores) + 1))
    scores.to_csv(OUTPUT_DIR / "validation_scores.csv", index=False)

    top_models = scores.nlargest(3, "validation_auc")["model"]
    ensemble_probability = np.mean([predictions[name] for name in top_models], axis=0)
    ensemble = sample.copy()
    ensemble["Exited"] = ensemble_probability
    ensemble.to_csv(OUTPUT_DIR / "ensemble_top3.csv", index=False)
    print(scores.to_string(index=False))
    print(f"\nGenerated {len(predictions) + 1} submission files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
