"""Evaluation helpers: classification report text."""
from sklearn.metrics import classification_report, confusion_matrix


def print_report(y_true, y_pred):
    print("\nConfusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_true, y_pred))
    print("\n", classification_report(y_true, y_pred, target_names=["Normal", "Failure soon"]))
