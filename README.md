# 🔧 AI-Powered Predictive Maintenance System for IoT Devices

Predict machine failures **hours before they happen** using IoT-style sensor data (temperature, vibration, current, pressure, RPM) and machine learning. Built with a **virtual IoT simulator**, so no hardware is needed.

![Failure prediction timeline](images/07_failure_prediction_timeline.png)

## 📌 Problem Statement
Reactive maintenance ("fix it when it breaks") causes unplanned downtime, safety risks and high repair cost. Predictive maintenance monitors sensor trends and raises an alert **before** failure so repairs can be scheduled.

## 🏭 Industry Relevance
Manufacturing plants, power plants, automotive, aviation and smart-factory platforms (Siemens, Bosch, GE, Schneider Electric, AWS IoT, Azure IoT) use this idea to cut downtime and maintenance cost.

## 🧰 Tech Stack
Python 3.10+, Pandas, NumPy, Scikit-learn (Logistic Regression, Random Forest), Matplotlib, Seaborn, Joblib, Flask (optional API).

## 📊 Dataset (Virtual Simulation)
`src/data_generator.py` simulates 40 machines (~35k readings, one every 10 min). Each machine runs healthy, then degrades (temperature, vibration and current rise; pressure and RPM fall) until failure. It includes daily load cycles, random harmless spikes, sensor noise and ~1% missing values.
Label `failure = 1` means failure occurs within the next 48 readings (~8 hours).
> The same pipeline works with real datasets (NASA C-MAPSS, AI4I 2020, Microsoft Azure PdM): replace `data/iot_sensor_data.csv` and adjust column names in `config.py`.

## 🏗️ Architecture
```
Virtual sensors -> Cleaning -> Feature engineering -> ML model -> Probability -> Alert -> Dashboard/plots
```
See `docs/PROJECT_GUIDE.md` for the detailed diagram.

## ⚙️ Installation
```bash
git clone https://github.com/<your-username>/AI-Predictive-Maintenance-IoT.git
cd AI-Predictive-Maintenance-IoT
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Usage
```bash
python main.py                  # full pipeline: data -> train -> evaluate -> plots
python -m src.simulate_live     # replay a machine as a live stream with alerts
python -m src.api               # optional Flask API on http://127.0.0.1:5000/predict
```
API example:
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
  -d '{"temperature":95,"vibration":6,"current":15,"pressure":85,"rpm":1400}'
```

## 📈 Results (test machines never seen in training)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.986 | 0.799 | 0.984 | 0.882 | 0.999 |
| Random Forest | 0.986 | 0.831 | 0.922 | 0.874 | 0.999 |

Alerts fired **7 to 13 hours before failure** on all 10 test machines (`outputs/alert_log.csv`).
Recall matters most here: a missed failure is costlier than a false alarm.
> Results come from synthetic data, so scores are optimistic compared with real plants.

## 🖼️ Screenshots
| Sensor trends | Confusion matrix | Feature importance |
|---|---|---|
| ![](images/03_sensor_trends.png) | ![](images/04_confusion_matrix.png) | ![](images/06_feature_importance.png) |

## 🎓 Learning Outcomes
Time-series feature engineering, handling imbalanced classes, leakage-free splitting by machine, evaluation beyond accuracy, alert design, simple model deployment with Flask.

## 🔮 Future Work
LSTM/GRU model, Remaining Useful Life regression, Streamlit dashboard, MQTT streaming, Docker, real NASA C-MAPSS data.

## 📄 License
MIT
