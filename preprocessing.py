"""Data loading and cleaning."""
import pandas as pd
from src import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by time, fill missing sensor values per machine, remove duplicates."""
    df = df.drop_duplicates(["machine_id", "timestamp"]).sort_values(["machine_id", "timestamp"]).copy()
    print("Missing values before cleaning:\n", df[config.SENSORS].isna().sum().to_string())
    # interpolate inside each machine so values never leak across machines
    df[config.SENSORS] = df.groupby("machine_id")[config.SENSORS].transform(
        lambda s: s.interpolate(limit_direction="both"))
    print("Missing values after cleaning :", int(df[config.SENSORS].isna().sum().sum()))
    return df.reset_index(drop=True)
