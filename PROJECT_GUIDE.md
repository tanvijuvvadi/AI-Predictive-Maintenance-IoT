# Project Guide (A to F)

## A. Project explanation
**Simple language:** Imagine a doctor who watches a machine's "vital signs" (heat, shaking, power use) and warns you *before* it falls sick. That is predictive maintenance.
**Technical language:** A supervised classification (or regression for Remaining Useful Life) problem on multivariate time-series. Sensor streams are cleaned, converted into rolling statistics, and fed to a classifier that outputs P(failure within horizon H). Thresholds on this probability produce alerts.

**Problem solved:** Reactive maintenance = unplanned downtime. Preventive (fixed schedule) maintenance = wasted parts. Predictive = repair only when needed.
**Industries:** manufacturing lines (motor/bearing failure), power plants (turbines, transformers), automotive (EV batteries, assembly robots), aviation (engine health monitoring), all with costly downtime.
**Benefits:** predict failures, reduce downtime, save repair cost, improve safety and efficiency.

**Workflow:** sensors -> collection (MQTT/HTTP) -> preprocessing -> feature engineering -> model training -> prediction -> alert -> dashboard.

## B. Tech stack options
| | A: Easiest | B: Intermediate | C: Advanced |
|---|---|---|---|
| Tools | Pandas, NumPy, Scikit-learn, Matplotlib | + Flask, Streamlit, Joblib | TensorFlow/PyTorch, MQTT, Docker, Kafka |
| Data | Simulated/CSV sensor data, AI4I 2020 | NASA C-MAPSS, Azure PdM | Streaming data, C-MAPSS RUL |
| Models | Logistic Regression, Random Forest | + XGBoost, Isolation Forest | LSTM, GRU, Transformer |
| Difficulty | Beginner | Intermediate | Advanced |
| GPU | No | No | Recommended |
| Output | Metrics + graphs + alerts | + live dashboard / API | + real-time streaming system |

## C. Selected approach
**Option A with a Flask API from Option B.** Easy to run on any laptop, every step explainable in interviews, and strong GitHub proof. LSTM is listed as future work.

## D. Architecture
```
 +----------------+   +-----------+   +--------------------+   +------------+
 | Virtual sensors|-->| Cleaning  |-->| Feature engineering|-->| ML model   |
 | (simulator)    |   | (fill NaN)|   | (rolling mean/std) |   | LR / RF    |
 +----------------+   +-----------+   +--------------------+   +-----+------+
                                                                      |
                     +--------------+   +-----------+                 v
                     | Plots / logs |<--| Alerts    |<-- P(failure) --+
                     +--------------+   | OK/WARN/CRIT|
                                        +-----------+
```
| Module | Role |
|---|---|
| data_generator.py | creates virtual IoT data |
| preprocessing.py | load, sort, interpolate missing values |
| features.py | rolling stats, differences, ratios |
| train.py | split by machine, train, compare, save model |
| evaluate.py | confusion matrix, classification report |
| predict.py | probability, alert levels, alert log |
| visualize.py | all graphs |
| simulate_live.py | stream replay with alerts |
| api.py | Flask prediction API |

## E. Folder structure
```
AI-Predictive-Maintenance-IoT/
├── data/        generated CSV dataset
├── notebooks/   optional Jupyter exploration
├── src/         all modular source code
├── models/      saved model (.pkl)
├── outputs/     metrics.json, alert_log.csv, predictions.csv
├── images/      graphs used in README
├── docs/        guides
├── README.md  requirements.txt  .gitignore  main.py
```

## Phase checklist (what, why, verify, common mistake)
| Phase | Do | Verify | Common mistake |
|---|---|---|---|
| 1 Setup | venv + pip install | `python -c "import sklearn"` | installing globally |
| 2 Dataset | run generator | CSV has ~35k rows | not setting seed |
| 3 Cleaning | interpolate per machine | missing = 0 | filling across machines |
| 4 Features | rolling mean/std | ~36 columns | rolling across machines |
| 5 Model | train LR + RF | metrics printed | random row split (leakage) |
| 6 Evaluation | precision, recall, F1, AUC | confusion matrix | trusting accuracy only |
| 7 Prediction | alert levels | alert_log.csv | no threshold reasoning |
| 8 Visualisation | run main.py | 8 PNGs in images/ | unlabeled axes |
| 9 GitHub | push + README | repo renders images | committing venv |
| 10 Final | screenshots, demo GIF | README complete | no results table |

## Mapping the original hardware plan to simulation
| Real hardware | Virtual equivalent |
|---|---|
| DHT22 temperature | `temperature` column |
| SW-420 vibration | `vibration` column (continuous, an accelerometer is more realistic) |
| ACS712 current | `current` (Amps) |
| Raspberry Pi loop | `simulate_live.py` |
| Flask on Pi/cloud | `api.py` |
| L298N motor shutdown | "motor OFF" print on CRITICAL |
Note: your notes label ADC values as "mA" and mention an LSTM saved as `.pkl`; raw ADC counts need calibration to Amps, and Keras LSTMs are saved as `.keras`. This repo avoids both issues.
