# 🔧 AI-Powered Predictive Maintenance System for IoT Devices

A machine-learning system that predicts equipment failures **about 8 hours before they happen**, using IoT-style sensor data (temperature, vibration, current, pressure, RPM). It runs on a **virtual IoT simulator**, so no physical hardware is needed.


## 📌 Overview
Factories lose money when machines break unexpectedly. This project learns the sensor patterns that appear before a machine fails, predicts the probability of failure, and raises WARNING/CRITICAL alerts so maintenance can be scheduled in time.

## ❓ Problem Statement
- **Reactive maintenance** (fix after failure) causes unplanned downtime and safety risks.
- **Scheduled maintenance** (fixed intervals) wastes parts and labour.
- **Predictive maintenance** repairs a machine only when its sensor data shows it is about to fail.

## 🏭 Industry Relevance
Used in manufacturing plants, power plants, automotive production and aviation engine monitoring. Companies such as Siemens, Bosch, GE and Schneider Electric, and platforms like AWS IoT and Azure IoT, offer predictive maintenance for smart factories (Industry 4.0).

## 🧰 Tech Stack
| Area | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| ML | Scikit-learn (Logistic Regression, Random Forest) |
| Visualization | Matplotlib, Seaborn |
| Deployment (optional) | Flask API |
| Environment | Google Colab / Jupyter / VS Code |

## 📊 Dataset (Virtual Simulation)
No real hardware was available, so a simulator generates the data:
- 40 virtual machines, about 35,000 readings (one every 10 minutes)
- Sensors: temperature, vibration, current, pressure, RPM
- Each machine runs healthy, degrades (heat, vibration and current rise; pressure and RPM fall), then fails
- Includes daily load cycles, sensor noise, harmless random spikes and about 1% missing values
- **Label:** `failure = 1` if the machine fails within the next 48 readings (about 8 hours)

> The pipeline also works with real datasets such as NASA C-MAPSS or AI4I 2020 by replacing the data file.

## 🏗️ Architecture

Virtual IoT sensors
        │
        ▼
 Data cleaning (fill missing values)
        │
        ▼
 Feature engineering (rolling mean/std, differences, ratios)
        │
        ▼
 ML model (Logistic Regression / Random Forest)
        │
        ▼
 Failure probability ──► Alert (OK / WARNING / CRITICAL)
        │
        ▼
 Graphs, alert log, live simulation
 ---

## 📁 Project Structure

AI-Predictive-Maintenance-IoT/
├── data/            # generated dataset
├── notebooks/       # Google Colab notebook
├── src/             # modular Python code
├── models/          # saved model
├── outputs/         # metrics, alert log, predictions
├── images/          # result graphs
├── docs/            # project and GitHub guides
├── README.md
├── requirements.txt
├── .gitignore
└── main.py

## ⚙️ Installation
bash
git clone https://github.com/your-username/AI-Predictive-Maintenance-IoT.git
cd AI-Predictive-Maintenance-IoT
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt


## ▶️ Usage
**Option 1: Google Colab (easiest).** Open the notebook in `notebooks/` and run all cells.

**Option 2: Run locally.**
bash
python main.py                  # data -> train -> evaluate -> graphs
python -m src.simulate_live     # live alert stream
python -m src.api               # optional Flask API

API test:
bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
  -d '{"temperature":95,"vibration":6,"current":15,"pressure":85,"rpm":1400}'

Response: `{"alert": "CRITICAL", "failure_probability": 1.0}`

## 📈 Results
Evaluated on 10 machines the model never saw during training (split by machine to avoid data leakage).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.986 | 0.799 | 0.984 | 0.882 | 0.999 |
| Random Forest | 0.986 | 0.831 | 0.922 | 0.874 | 0.999 |

- **Recall of 98%:** almost every upcoming failure was caught
- **Average warning lead time: about 10 hours** before failure
- Recall is the key metric, because a missed failure costs far more than a false alarm
- Results come from synthetic data, so real-world scores would be lower


## 🎓 Learning Outcomes
- Simulating IoT sensor data and failure behaviour
- Cleaning time-series data and handling missing values
- Feature engineering with rolling statistics
- Handling imbalanced classes
- Preventing data leakage by splitting on machines
- Evaluating with precision, recall, F1 and ROC-AUC
- Designing an alert system and serving predictions with Flask

## 🔮 Future Improvements
- LSTM/GRU deep-learning model
- Remaining Useful Life (RUL) prediction
- Streamlit dashboard
- Real-time streaming with MQTT
- Training on NASA C-MAPSS data
- Docker deployment



