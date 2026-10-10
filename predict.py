"""Prediction + alert generation."""
import joblib
import pandas as pd
from src import config


def load_model():
    bundle = joblib.load(config.MODEL_PATH)
    return bundle["model"], bundle["features"]


def alert_level(prob: float) -> str:
    if prob >= config.CRITICAL_THRESHOLD:
        return "CRITICAL"
    if prob >= config.WARNING_THRESHOLD:
        return "WARNING"
    return "OK"


def add_predictions(df: pd.DataFrame, model, feats) -> pd.DataFrame:
    df = df.copy()
    df["failure_prob"] = model.predict_proba(df[feats])[:, 1]
    df["alert"] = df["failure_prob"].apply(alert_level)
    return df


def build_alert_log(df: pd.DataFrame) -> pd.DataFrame:
    """First WARNING per machine and how many hours before the failure it appeared."""
    rows = []
    for mid, g in df.groupby("machine_id"):
        warn = g[g["alert"] != "OK"]
        if len(warn):
            first = warn.iloc[0]
            lead_h = first["steps_to_failure"] * config.SAMPLING_MINUTES / 60
            rows.append({"machine_id": mid, "first_alert_time": first["timestamp"],
                         "level": first["alert"], "hours_before_failure": round(lead_h, 1)})
        else:
            rows.append({"machine_id": mid, "first_alert_time": None, "level": "MISSED",
                         "hours_before_failure": 0})
    return pd.DataFrame(rows)
