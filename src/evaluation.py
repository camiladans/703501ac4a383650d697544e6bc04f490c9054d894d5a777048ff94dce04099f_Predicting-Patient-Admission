"""
Module: evaluation.py

Evaluates a classification model on test data.
Focuses on inpatient prediction ('IN') using recall and PR AUC.
Generates SHAP and confusion matrix visualizations.
"""

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

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
    report_path: str = "reports/metrics.txt",
    shap_path: str = "reports/shap_summary.html"
):
    """
    Evaluates model performance on test data and saves metrics and visualizations.

    Args:
        model: Trained classifier.
        test_data (pd.DataFrame): DataFrame with features and 'SOURCE' target.
        report_path (str): Path to save metrics report.
        shap_path (str): Path to save SHAP force plot.

    Returns:
        dict: Dictionary with evaluation metrics.
    """
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

    # Confusion matrix plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["OUT", "IN"], yticklabels=["OUT", "IN"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved to: reports/confusion_matrix.png")

    # SHAP force plot
    try:
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)
        shap_html = shap.plots.force(shap_values[0], matplotlib=False)
        with open(shap_path, "w") as f:
            f.write(shap.getjs())
            f.write(shap_html.html())
        print(f"SHAP summary saved to: {shap_path}")
    except Exception as e:
        print(f"⚠️ SHAP explanation failed: {str(e)}")

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
