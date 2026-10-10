"""
Virtual IoT simulator.
Generates realistic sensor streams for many virtual machines. Each machine runs healthy,
then degrades (heat, vibration and current rise, pressure and speed drop) and finally fails.
This replaces real hardware (Raspberry Pi + DHT22 + SW-420 + ACS712).
"""
import numpy as np
import pandas as pd
from src import config


def simulate_machine(machine_id: int, rng: np.random.Generator) -> pd.DataFrame:
    life = int(rng.integers(600, 1200))                 # total readings until failure
    degr_len = int(rng.integers(90, 160))               # how long the wear phase lasts
    t = np.arange(life)
    steps_to_failure = life - 1 - t

    # severity: 0 = healthy, 1 = about to fail (accelerating curve)
    progress = np.clip((t - (life - degr_len)) / degr_len, 0, 1)
    severity = progress ** 1.8

    # machine-to-machine baseline differences + daily load cycle
    base = {
        "temperature": rng.normal(62, 3), "vibration": rng.normal(2.0, 0.2),
        "current": rng.normal(10, 0.8), "pressure": rng.normal(100, 4), "rpm": rng.normal(1500, 30),
    }
    daily = np.sin(2 * np.pi * t / 144)                 # 144 readings = 24 h

    df = pd.DataFrame({
        "machine_id": machine_id,
        "timestamp": pd.Timestamp("2025-01-01") + pd.to_timedelta(t * config.SAMPLING_MINUTES, unit="m"),
        "temperature": base["temperature"] + 2.0 * daily + 28 * severity + rng.normal(0, 1.2, life),
        "vibration": base["vibration"] + 0.2 * daily + 4.5 * severity + rng.normal(0, 0.25, life),
        "current": base["current"] + 0.5 * daily + 5.0 * severity + rng.normal(0, 0.5, life),
        "pressure": base["pressure"] - 18 * severity + rng.normal(0, 2.0, life),
        "rpm": base["rpm"] - 120 * severity + rng.normal(0, 15, life),
    })

    # harmless random spikes (so the model must not just react to every spike)
    spikes = rng.random(life) < 0.01
    df.loc[spikes, "vibration"] += rng.uniform(1.5, 3.0, spikes.sum())

    df["steps_to_failure"] = steps_to_failure
    df["failure"] = (steps_to_failure <= config.HORIZON).astype(int)
    return df


def generate_dataset(n_machines: int = config.N_MACHINES, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.concat([simulate_machine(i + 1, rng) for i in range(n_machines)], ignore_index=True)

    # simulate sensor dropouts (~1% missing values) like real IoT networks
    for col in config.SENSORS:
        mask = rng.random(len(df)) < 0.01
        df.loc[mask, col] = np.nan
    return df


if __name__ == "__main__":
    data = generate_dataset()
    config.DATA_PATH.parent.mkdir(exist_ok=True)
    data.to_csv(config.DATA_PATH, index=False)
    print(f"Saved {len(data):,} rows to {config.DATA_PATH}")
    print(data.head())
