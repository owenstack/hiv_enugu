import pandas as pd
import numpy as np
import warnings
import joblib
import os
from loguru import logger
from hiv_enugu.modeling.data_prep import prepare_data_for_modeling
from hiv_enugu.modeling.fit import fit_growth_models
from hiv_enugu.modeling.ensemble import build_ensemble_models
from hiv_enugu.modeling.features import create_ml_features
from hiv_enugu.config import PROCESSED_DATA_DIR, FIGURES_DIR, RAW_DATA_DIR
from hiv_enugu.plotting.evaluation import (
    visualize_individual_models,
    visualize_ensemble_comparison,
    visualize_metrics_comparison,
    create_validation_plot,
    forecast_future_trends
)
from hiv_enugu.plotting.diagnostics import plot_residuals, plot_residuals_histogram
from hiv_enugu.data_processing import load_data
from hiv_enugu.utils import generate_bootstrap_predictions

warnings.filterwarnings("ignore")


def main():
    logger.info("Starting HIV Modeling Training Pipeline...")
    file_path = PROCESSED_DATA_DIR / "cleaned_enrollments.csv"
    # 1. Load and Prepare Data
    weekly_df = load_data(file_path)
    if weekly_df is None:
        logger.error("Failed to load data. Exiting.")
        return
    X, y, cv_splits, tscv = prepare_data_for_modeling(weekly_df)

    # Use last split for visualization
    final_train_idx, final_test_idx = cv_splits[-1]
    X_train, X_test = X[final_train_idx], X[final_test_idx]
    y_train, y_test = y[final_train_idx], y[final_test_idx]

    # 2. Fit Individual Growth Models
    fitted_models, model_metrics = fit_growth_models(X, y, cv_splits)

    # Visualize individual model fits
    visualize_individual_models(
        weekly_df,
        X,
        final_train_idx,
        final_test_idx,
        y_train,
        y_test,
        fitted_models,
        filename="individual_model_fits.png"
    )

    # 3. Build Ensemble Models
    ensemble_models, ensemble_metrics = build_ensemble_models(X, y, fitted_models, model_metrics, cv_splits)

    # Identify best models
    best_model_name = max(model_metrics, key=lambda k: model_metrics[k]["test_r2"])
    best_model = fitted_models[best_model_name]

    best_ensemble_name = max(ensemble_metrics, key=lambda k: ensemble_metrics[k]["r2"])
    best_ensemble_model = ensemble_models[best_ensemble_name]

    logger.info(
        f"\nBest Individual Model: {best_model_name} (R2: {model_metrics[best_model_name]['test_r2']:.4f})"
    )
    logger.info(
        f"Best Ensemble Model: {best_ensemble_name} (R2: {ensemble_metrics[best_ensemble_name]['r2']:.4f})"
    )

    # 4. Visualize Ensemble Comparison and Metrics
    visualize_ensemble_comparison(
        weekly_df,
        X,
        y,
        cv_splits,
        fitted_models,
        ensemble_models,
        best_model_name,
        filename="ensemble_comparison.png",
    )
    visualize_metrics_comparison(
        model_metrics, ensemble_metrics, filename="model_metrics_comparison.png"
    )
    create_validation_plot(
        weekly_df,
        X,
        y,
        best_model,
        best_ensemble_model,
        best_model_name,
        best_ensemble_name,
        filename="best_model_vs_ensemble.png",
    )

    # 5. Diagnostic Plots for Best Ensemble Model
    y_pred_ensemble = best_ensemble_model["predict"](X)
    plot_residuals(
        y,
        y_pred_ensemble,
        best_ensemble_name,
        filename="residuals_plot_ensemble.png",
    )
    plot_residuals_histogram(
        y,
        y_pred_ensemble,
        best_ensemble_name,
        filename="residuals_histogram_ensemble.png",
    )

    # 6. Forecast Future Trends
    forecast_future_trends(
        weekly_df,
        X,
        y,
        best_model,
        best_ensemble_model,
        best_ensemble_name,
        fitted_models,
        best_model,
        generate_bootstrap_predictions,
        filename="future_forecast.png",
    )

    logger.info("\nAnalysis complete. All visualizations saved to reports/figures/")


if __name__ == "__main__":
    main()
