# GitHub Guide (J, L, M)

## J. Upload strategy
**Repo name:** `AI-Predictive-Maintenance-IoT`
**Description:** Machine-learning system that predicts equipment failures from simulated IoT sensor data and raises early alerts.
**Topics:** `predictive-maintenance` `iot` `machine-learning` `python` `scikit-learn` `time-series` `industry-4-0` `data-science` `random-forest` `flask`

```bash
cd AI-Predictive-Maintenance-IoT
git init
git branch -M main
git add .
git commit -m "Initial project structure"
git remote add origin https://github.com/<your-username>/AI-Predictive-Maintenance-IoT.git
git push -u origin main
```
Create the empty repo on github.com first (no README). Keep `images/` committed so README pictures show. Add a license, pin the repo on your profile, and write a clean commit message every time (`feat:`, `docs:`, `fix:`).

## L. 7-day commit plan
| Day | Work | Commit messages | Proof |
|---|---|---|---|
| 1 | Setup, folders, requirements, .gitignore | `chore: project structure and environment` | repo tree screenshot |
| 2 | Data generator + dataset | `feat: virtual IoT data simulator` | dataset preview (`df.head()`) |
| 3 | Cleaning + features | `feat: preprocessing`, `feat: rolling features` | missing-value before/after, 01-03 PNGs |
| 4 | Training | `feat: train LR and RF with machine-wise split` | training log screenshot |
| 5 | Evaluation | `feat: metrics, confusion matrix, ROC` | 04, 05, 08 PNGs, metrics.json |
| 6 | Alerts, live sim, plots, API | `feat: alert system and live simulation`, `feat: flask api` | 07 timeline, terminal GIF |
| 7 | README, docs, polish | `docs: README with results`, `docs: guides` | repo preview, pinned repo |

## M. Proof checklist
- [ ] Dataset preview (`images/` or README)
- [ ] Preprocessing output (missing values before/after)
- [ ] Training log (terminal screenshot)
- [ ] Prediction output (`outputs/predictions.csv`, `alert_log.csv`)
- [ ] Failure detection graph (`07_failure_prediction_timeline.png`)
- [ ] Confusion matrix (`04_confusion_matrix.png`)
- [ ] Final output: live simulation terminal recording
- [ ] GitHub repo preview with README rendered
- [ ] 8+ meaningful commits across several days
- [ ] LinkedIn post linking the repo

**Naming:** `NN_descriptive_name.png`, saved in `images/`, referenced in README with `![alt](images/file.png)`.

## Interview talking points
Why split by machine (leakage)? Why recall over accuracy? Why class weights? How would you move to real data or an LSTM? How would the alert threshold be tuned using the cost of a missed failure vs a false alarm?
