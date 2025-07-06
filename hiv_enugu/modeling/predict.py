import pandas as pd
import numpy as np
import joblib
import os
from hiv_enugu.modeling.features import create_ml_features
from hiv_enugu.config import PROCESSED_DATA_DIR, MODELS_DIR
from hiv_enugu.data_processing import load_data

# Define paths
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_enrollments.csv"
FORECAST_OUTPUT_PATH = PROCESSED_DATA_DIR / "forecast_results_2024_2028.csv"


def load_models():
    """Loads the trained Random Forest, Gradient Boosting models and the scaler."""
    try:
        rf_model = joblib.load(f"{MODELS_DIR}/rf_model.pkl")
        gb_model = joblib.load(f"{MODELS_DIR}/gb_model.pkl")
        scaler = joblib.load(f"{MODELS_DIR}/feature_scaler.pkl")
        print("Models and scaler loaded successfully.")
        return rf_model, gb_model, scaler
    except FileNotFoundError:
        print("Error: Trained models or scaler not found. Please run train.py first.")
        return None, None, None


def generate_predictions(df, rf_model, gb_model, scaler, fitted_models):
    """Generates predictions using the best ensemble model."""
    X_full = df["time_idx"].values

    # Create features for the ML models
    ml_features = create_ml_features(X_full, fitted_models, X_full)
    ml_features_scaled = scaler.transform(ml_features)

    # For simplicity, let's use the RF model as the best ensemble for prediction here.
    # In a real scenario, you'd select the best based on evaluation metrics.
    predictions = rf_model.predict(ml_features_scaled)
    return predictions


def main():
    print("Starting HIV Prediction Pipeline...")

    # 1. Load Data
    df = load_data(PROCESSED_DATA_PATH)
    if df is None:
        print("Failed to load data. Exiting.")
        return

    # 2. Load Models
    rf_model, gb_model, scaler = load_models()
    if rf_model is None:
        return

    # Placeholder for fitted_models (needed for create_ml_features)
    # In a real scenario, these would be loaded or passed from training.
    # For now, we'll create dummy ones or load if available.
    # This part needs to be aligned with how fitted_models are saved/loaded.
    fitted_models = {}
    try:
        feature_scaler = joblib.load(f"{MODELS_DIR}/feature_scaler.pkl")
        gb_model = joblib.load(f"{MODELS_DIR}/gb_model.pkl")
        rf_model = joblib.load(f"{MODELS_DIR}/rf_model.pkl")
    except FileNotFoundError:
        print("Warning: Fitted growth models not found. Predictions might be inaccurate.")
        # Fallback or error handling if growth models are crucial for feature creation

    # 3. Generate Predictions
    # This part needs to be refined based on how the best ensemble model is determined and saved.
    # For now, using RF model directly.
    predictions = generate_predictions(df, rf_model, gb_model, scaler, fitted_models)
    df["predicted_cumulative"] = predictions

    # 4. Save Predictions
    os.makedirs(os.path.dirname(FORECAST_OUTPUT_PATH), exist_ok=True)
    df.to_csv(FORECAST_OUTPUT_PATH, index=False)
    print(f"Predictions saved to {FORECAST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
