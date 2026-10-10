"""Central configuration. Change values here, not inside other files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "iot_sensor_data.csv"
MODEL_PATH = ROOT / "models" / "predictive_maintenance_model.pkl"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_DIR = ROOT / "images"

RANDOM_STATE = 42
N_MACHINES = 40              # number of virtual machines
SAMPLING_MINUTES = 10        # one sensor reading every 10 minutes
HORIZON = 48                 # label = 1 if failure happens within next 48 readings (~8 hours)

SENSORS = ["temperature", "vibration", "current", "pressure", "rpm"]
ROLLING_WINDOWS = [6, 18]    # 1 hour and 3 hours

WARNING_THRESHOLD = 0.50     # failure probability -> WARNING alert
CRITICAL_THRESHOLD = 0.80    # failure probability -> CRITICAL alert
