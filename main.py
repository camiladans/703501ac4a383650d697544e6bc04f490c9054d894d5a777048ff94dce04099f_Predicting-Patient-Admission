"""
Main pipeline script for predicting inpatient vs. outpatient status.
Performs data loading, preprocessing, training, evaluation, and SHAP explainability.
"""

from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.model_training import train_model, save_model
from src.evaluation import evaluate_model

def main():
    print("Hello from 703501ac4a383650d697544e6bc04f490c9054d894d5a777048ff94dce04099f-predicting-patient-admission!")
    print("Starting patient admission prediction pipeline...")

    # Load and preprocess
    train_data, test_data = preprocess_data("data/raw/data-ori.xlsx")

    # Feature engineering
    train_data = engineer_features(train_data)
    test_data = engineer_features(test_data)

    # Train model
    model, metrics = train_model(train_data, model_type="logreg")

    # Save model
    save_model(model, "models/model.pkl")

    # Evaluate model
    evaluate_model(model, test_data, report_path="reports/metrics.txt", shap_path="reports/shap_summary.html")

    print("SUCCESS! - Pipeline completed.")

if __name__ == "__main__":
    main()
