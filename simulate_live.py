"""Replay a machine as a live IoT stream and print alerts (virtual Raspberry Pi loop)."""
import sys
import time
from src import config
from src.predict import load_model, add_predictions
from src.preprocessing import load_data, clean_data
from src.features import add_features


def run(machine_id=None, delay=0.0, every=12):
    model, feats = load_model()
    df = add_predictions(add_features(clean_data(load_data())), model, feats)
    machine_id = machine_id or int(df["machine_id"].iloc[-1])
    g = df[df.machine_id == machine_id]
    print(f"Streaming machine {machine_id} ({len(g)} readings)...\n")
    for _, r in g.iloc[::every].iterrows():
        icon = {"OK": "✅", "WARNING": "⚠️ ", "CRITICAL": "🚨"}[r["alert"]]
        print(f"{r['timestamp']}  T={r['temperature']:.1f}C  V={r['vibration']:.2f}  "
              f"I={r['current']:.1f}A  P(fail)={r['failure_prob']:.2f}  {icon} {r['alert']}")
        if r["alert"] == "CRITICAL":
            print("   -> Action: motor OFF + maintenance ticket created (simulated)")
        time.sleep(delay)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else None)
