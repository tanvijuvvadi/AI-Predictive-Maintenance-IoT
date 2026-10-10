"""All plots saved to images/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve
from src import config

config.IMAGE_DIR.mkdir(exist_ok=True)


def plot_sensor_trends(df, machine_id):
    g = df[df.machine_id == machine_id]
    fig, axes = plt.subplots(len(config.SENSORS), 1, figsize=(11, 10), sharex=True)
    for ax, col in zip(axes, config.SENSORS):
        ax.plot(g["timestamp"], g[col], lw=0.8)
        ax.fill_between(g["timestamp"], g[col].min(), g[col].max(), where=g["failure"] == 1,
                        color="red", alpha=0.15, label="Failure zone (next 8 h)")
        ax.set_ylabel(col)
    axes[0].legend(loc="upper left")
    axes[0].set_title(f"Machine {machine_id}: sensor trends before failure")
    plt.tight_layout(); plt.savefig(config.IMAGE_DIR / "03_sensor_trends.png", dpi=150); plt.close()


def plot_class_balance(df):
    plt.figure(figsize=(5, 4))
    sns.countplot(x="failure", data=df)
    plt.xticks([0, 1], ["Normal", "Failure soon"]); plt.title("Class distribution")
    plt.tight_layout(); plt.savefig(config.IMAGE_DIR / "01_class_balance.png", dpi=150); plt.close()


def plot_correlation(df):
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(df[config.SENSORS + ["failure"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Sensor correlation"); plt.tight_layout()
    plt.savefig(config.IMAGE_DIR / "02_correlation.png", dpi=150); plt.close()


def plot_confusion(y_true, y_pred):
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Failure"], yticklabels=["Normal", "Failure"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix")
    plt.tight_layout(); plt.savefig(config.IMAGE_DIR / "04_confusion_matrix.png", dpi=150); plt.close()


def plot_roc(y_true, prob):
    fpr, tpr, _ = roc_curve(y_true, prob)
    plt.figure(figsize=(5, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--", c="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curve")
    plt.tight_layout(); plt.savefig(config.IMAGE_DIR / "05_roc_curve.png", dpi=150); plt.close()


def plot_feature_importance(model, feats, top=15):
    clf = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(clf, "feature_importances_"):          # Random Forest
        values = clf.feature_importances_
    else:                                              # Logistic Regression -> |coefficients|
        values = abs(clf.coef_[0])
    imp = pd.Series(values, index=feats).nlargest(top)[::-1]
    plt.figure(figsize=(7, 5)); imp.plot.barh(); plt.title("Top feature importances")
    plt.tight_layout(); plt.savefig(config.IMAGE_DIR / "06_feature_importance.png", dpi=150); plt.close()


def plot_failure_timeline(df, machine_id):
    g = df[df.machine_id == machine_id]
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(g["timestamp"], g["temperature"], label="temperature")
    ax[0].plot(g["timestamp"], g["vibration"] * 10, label="vibration x10")
    ax[0].legend(); ax[0].set_title(f"Machine {machine_id}: prediction timeline")
    ax[1].plot(g["timestamp"], g["failure_prob"], c="purple", label="predicted failure probability")
    ax[1].axhline(config.WARNING_THRESHOLD, ls="--", c="orange", label="WARNING")
    ax[1].axhline(config.CRITICAL_THRESHOLD, ls="--", c="red", label="CRITICAL")
    ax[1].fill_between(g["timestamp"], 0, 1, where=g["failure"] == 1, color="red", alpha=0.12,
                       label="True failure zone")
    ax[1].legend(loc="upper left"); ax[1].set_ylabel("probability")
    plt.tight_layout(); plt.savefig(config.IMAGE_DIR / "07_failure_prediction_timeline.png", dpi=150); plt.close()


def plot_model_comparison(results):
    df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    df.plot.bar(figsize=(8, 4.5), rot=0); plt.ylim(0, 1.05); plt.title("Model comparison")
    plt.tight_layout(); plt.savefig(config.IMAGE_DIR / "08_model_comparison.png", dpi=150); plt.close()
