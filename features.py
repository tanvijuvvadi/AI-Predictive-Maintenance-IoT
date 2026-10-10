"""Feature engineering: turn raw readings into trends that reveal wear."""
import pandas as pd
from src import config


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["machine_id", "timestamp"]).copy()
    g = df.groupby("machine_id")
    for col in config.SENSORS:
        for w in config.ROLLING_WINDOWS:
            df[f"{col}_mean_{w}"] = g[col].transform(lambda s: s.rolling(w, min_periods=1).mean())
            df[f"{col}_std_{w}"] = g[col].transform(lambda s: s.rolling(w, min_periods=1).std()).fillna(0)
        df[f"{col}_diff"] = g[col].diff().fillna(0)
    df["power_proxy"] = df["current"] * df["rpm"] / 1000      # rough load indicator
    df["heat_vib_ratio"] = df["temperature"] / (df["vibration"] + 1e-6)
    return df


def get_feature_columns(df: pd.DataFrame):
    drop = {"machine_id", "timestamp", "failure", "steps_to_failure"}
    return [c for c in df.columns if c not in drop]
