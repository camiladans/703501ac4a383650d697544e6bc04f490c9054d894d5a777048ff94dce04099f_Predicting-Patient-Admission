"""
Module: evaluation.py

Evaluates a classification model on test data.
Focuses on inpatient prediction ('IN') using recall and PR AUC.
"""

import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

def evaluate_model(
    model,
    test_data: pd.DataFrame,
    report_path: str = "reports/metrics.txt"
):
    """
    Evaluates model performance on test data and saves metrics.

    Args:
        model: Trained classifier.
        test_data (pd.DataFrame): DataFrame with features and 'SOURCE' target.
        report_path (str): Path to save metrics report.

    Returns:
        dict: Dictionary with evaluation metrics.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    if "SOURCE" not in test_data.columns:
        raise KeyError("The column 'SOURCE' is missing from the input DataFrame.")

    # Binary encode target: IN = 1 (inpatient), OUT = 0
    y_test = test_data["SOURCE"].apply(lambda x: 1 if str(x).upper() == "IN" else 0)
    X_test = test_data.drop(columns=["SOURCE"])

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["OUT", "IN"])

    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    pr_auc = average_precision_score(y_test, y_proba) if y_proba is not None else None

    # Save metrics to file
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n\n")
        f.write("Classification Report:\n" + report + "\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n")
        if roc_auc:
            f.write(f"ROC AUC: {roc_auc:.4f}\n")
        if pr_auc:
            f.write(f"PR AUC: {pr_auc:.4f}\n")

    print(f"Metrics saved to: {report_path}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

def load_model(filepath: str = "models/model.pkl"):
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model
