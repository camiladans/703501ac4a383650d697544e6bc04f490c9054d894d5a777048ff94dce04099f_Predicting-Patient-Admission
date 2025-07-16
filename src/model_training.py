"""
Module: model_training.py

Trains a classification model (LogReg or RF) on processed data.
Prioritizes recall for inpatients ('in').
"""

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

def train_model(train_data: pd.DataFrame, model_type: str = "logreg"):
    print("Train data columns:", train_data.columns.tolist())
    if "SOURCE" not in train_data.columns:
         raise KeyError("The column 'SOURCE' is missing from the input DataFrame.")

    y = train_data["SOURCE"].apply(lambda x: 1 if str(x).upper() == "IN" else 0)  # Split features and target
    X = train_data.drop(columns=["SOURCE"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        param_grid = {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}
    elif model_type == "rf":
        model = RandomForestClassifier(random_state=42, class_weight="balanced")
        param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
    else:
        raise ValueError("Unsupported model_type. Choose 'logreg' or 'rf'.")

    grid = GridSearchCV(model, param_grid, cv=5, scoring='recall')
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_val)

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "report": classification_report(y_val, y_pred)
    }

    return best_model, metrics

def save_model(model, filepath: str = "models/model.pkl"):
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")
