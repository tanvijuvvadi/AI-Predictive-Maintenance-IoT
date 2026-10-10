"""Optional Flask API: POST sensor readings, get failure probability.
Run: python -m src.api   (uses raw sensors only, so features are approximated by current values)."""
from flask import Flask, request, jsonify
import pandas as pd
from src.predict import load_model, alert_level

app = Flask(__name__)
model, feats = load_model()


@app.route("/predict", methods=["POST"])
def predict():
    d = request.get_json()
    # Fill any engineered feature from the raw value it was built from (single-reading approximation)
    row = {}
    for f in feats:
        base = next((s for s in ["temperature", "vibration", "current", "pressure", "rpm"] if f.startswith(s)), None)
        row[f] = 0.0 if f.endswith("_diff") or "_std_" in f else float(d.get(base, 0)) if base else 0.0
    row["power_proxy"] = d["current"] * d["rpm"] / 1000
    row["heat_vib_ratio"] = d["temperature"] / (d["vibration"] + 1e-6)
    prob = float(model.predict_proba(pd.DataFrame([row])[feats])[0, 1])
    return jsonify({"failure_probability": round(prob, 3), "alert": alert_level(prob)})


if __name__ == "__main__":
    app.run(port=5000)
