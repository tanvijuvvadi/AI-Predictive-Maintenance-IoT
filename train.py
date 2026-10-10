"""Train and compare models. Split is by MACHINE to avoid data leakage."""
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from src import config
from src.features import get_feature_columns


def split_by_machine(df):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df["machine_id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def train_models(train_df, test_df):
    feats = get_feature_columns(train_df)
    X_tr, y_tr, X_te, y_te = train_df[feats], train_df["failure"], test_df[feats], test_df["failure"]

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                                class_weight="balanced", n_jobs=-1,
                                                random_state=config.RANDOM_STATE),
    }
    results, fitted = {}, {}
    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]
        results[name] = compute_metrics(y_te, (prob >= 0.5).astype(int), prob)
        fitted[name] = model
        print("  ", {k: round(v, 4) for k, v in results[name].items()})

    best = max(results, key=lambda n: results[n]["f1"])
    print(f"\nBest model: {best}")
    config.MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": fitted[best], "features": feats, "name": best}, config.MODEL_PATH)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    (config.OUTPUT_DIR / "metrics.json").write_text(json.dumps({"best": best, "results": results}, indent=2))
    return fitted[best], best, feats, results
