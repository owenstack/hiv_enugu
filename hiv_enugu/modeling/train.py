import pandas as pd
import numpy as np
import warnings
import joblib
import os
from loguru import logger
from hiv_enugu.modeling.data_prep import load_and_prepare_data_for_modeling, get_cv_splits
from hiv_enugu.modeling.fit import fit_individual_models
from hiv_enugu.modeling.ensemble import build_ensemble_models
from hiv_enugu.modeling.features import create_ml_features
from hiv_enugu.config import PROCESSED_DATA_DIR, FIGURES_DIR
from hiv_enugu.plotting.evaluation import (
    visualize_individual_models,
    visualize_ensemble_comparison,
    visualize_metrics_comparison,
    plot_best_model_vs_ensemble,
    plot_forecast,
)
from hiv_enugu.plotting.diagnostics import plot_residuals, plot_residuals_histogram

warnings.filterwarnings("ignore")


def main():
    logger.info("Starting HIV Modeling Training Pipeline...")

    # 1. Load and Prepare Data
    df = load_and_prepare_data_for_modeling()
    if df is None:
        logger.error("Failed to load data. Exiting.")
        return

    X = df["time_idx"].values
    y = df["cumulative"].values
    cv_splits = get_cv_splits(df, n_splits=5)

    # Use last split for visualization
    final_train_idx, final_test_idx = cv_splits[-1]
    X_train, X_test = X[final_train_idx], X[final_test_idx]
    y_train, y_test = y[final_train_idx], y[final_test_idx]

    # 2. Fit Individual Growth Models
    fitted_models, model_metrics = fit_individual_models(X, y, cv_splits)

    # Visualize individual model fits
    visualize_individual_models(
        df,
        y,
        fitted_models,
        cv_splits,
        filename="individual_model_fits.png",
    )

    # 3. Build Ensemble Models
    ensemble_models, ensemble_metrics = build_ensemble_models(
        X, y, fitted_models, model_metrics, cv_splits
    )

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
        df,
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
    plot_best_model_vs_ensemble(
        df,
        y,
        X,
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
    plot_forecast(
        df,
        y,
        fitted_models,
        best_model,
        best_ensemble_model,
        best_ensemble_name,
        filename="future_forecast.png",
    )

    logger.info("\nAnalysis complete. All visualizations saved to reports/figures/")


if __name__ == "__main__":
    main()