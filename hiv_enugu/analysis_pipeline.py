import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

import joblib
from loguru import logger
import numpy as np
import pandas as pd

from hiv_enugu.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

# Custom module imports
from hiv_enugu.data_processing import load_data
from hiv_enugu.modeling.data_prep import prepare_data_for_modeling
from hiv_enugu.modeling.ensemble import build_ensemble_models_with_cv
from hiv_enugu.modeling.features import create_ml_features
from hiv_enugu.modeling.fit import fit_growth_models
from hiv_enugu.plotting.diagnostics import plot_qq, plot_residuals, plot_residuals_histogram
from hiv_enugu.plotting.evaluation import (  # Ensure plot_feature_importances is imported
    create_validation_plot,
    forecast_future_trends,
    plot_feature_importances,  # New
    visualize_ensemble_comparison,
    visualize_individual_models,
    visualize_metrics_comparison,
    visualize_single_model_fit,
    visualize_weighted_average_metrics_comparison,  # Added import
)
from hiv_enugu.reporting import (
    generate_coefficient_table,
    generate_ensemble_input_tables,
    generate_feature_importance_table,  # New
    generate_full_predictions_table,
    generate_predictions_summary_table,
    generate_standalone_metrics_table,
    generate_weighted_average_comparison_table,
    generate_weighted_parameters_table,
)
from hiv_enugu.utils import (
    generate_bootstrap_predictions,
)  # generate_bootstrap_predictions type: Callable

warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore")

# Type Aliases
ModelType = Any  # General placeholder for any model type (e.g., scikit-learn estimator)
ScalerType = Any  # General placeholder for a scaler (e.g., StandardScaler)
CVObject = Any  # Placeholder for TimeSeriesSplit object or similar

# --- Global State ---
# These globals store results from the training phase for potential use in a subsequent prediction phase
# if run within the same session, or as a cache.
# For robust standalone prediction, saving/loading artifacts to/from disk is preferred.
fitted_models_global: Dict[str, ModelType] = {}
model_metrics_global: Dict[str, Dict[str, float]] = {}
best_model_name_global: Optional[str] = None
best_ensemble_model_name_global: Optional[str] = None
# X_global stores the 'time_idx' array from the most recent training data.
# Crucial for `create_ml_features` if growth models are used and need original fitting context.
X_global: Optional[np.ndarray] = None


# --- Helper Functions ---


def _validate_dataframe_for_training(
    df: Optional[pd.DataFrame], df_name: str = "DataFrame"
) -> bool:
    """Validates if the DataFrame is suitable for training."""
    if df is None or df.empty:
        logger.error(f"{df_name} is None or empty. Cannot proceed with training.")
        return False
    if not isinstance(df, pd.DataFrame):
        logger.error(f"{df_name} is not a Pandas DataFrame. Cannot proceed.")
        return False
    required_cols = ["time_idx", "cumulative"]
    if not all(col in df.columns for col in required_cols):
        logger.error(
            f"Error: '{df_name}' must contain {required_cols} columns for training. Found: {df.columns.tolist()}."
        )
        return False
    return True


def _prepare_train_test_split_from_cv(
    X: np.ndarray, y: np.ndarray, cv_splits: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts training/testing sets from the last CV split, handling empty or out-of-bounds cases."""
    final_train_idx, final_test_idx = cv_splits[-1] if cv_splits else (np.array([]), np.array([]))

    def get_slice(
        data_array: np.ndarray, indices: np.ndarray, default_array: np.ndarray
    ) -> np.ndarray:
        if indices.size > 0 and data_array.size > 0 and indices.max() < data_array.shape[0]:
            return data_array[indices]
        return default_array

    X_train = get_slice(X, final_train_idx, X)  # Default to full X if no valid train indices
    y_train = get_slice(y, final_train_idx, y)  # Default to full y
    X_test = get_slice(
        X, final_test_idx, np.array([])
    )  # Default to empty if no valid test indices
    y_test = get_slice(y, final_test_idx, np.array([]))

    return X_train, y_train, X_test, y_test, final_train_idx, final_test_idx


def _determine_best_model(
    metrics_dict: Dict[str, Dict[str, float]],
    model_type_name: str,
    primary_metric_key: str = "test_r2",
    secondary_metric_key: Optional[str] = "r2",
) -> Optional[str]:
    """Determines the best model from metrics, trying primary then secondary keys."""
    if not metrics_dict:
        logger.warning(
            f"Cannot determine best {model_type_name} model: Metrics dictionary is empty."
        )
        return None

    best_name: Optional[str] = None
    best_score: float = -float("inf")  # Initialize with negative infinity
    used_key: str = primary_metric_key

    # Try primary key
    for name, metrics in metrics_dict.items():
        score = metrics.get(primary_metric_key, -float("inf"))
        if score > best_score:
            best_score = score
            best_name = name

    # If primary key yielded no valid model or score, try secondary key
    if secondary_metric_key and (best_name is None or best_score == -float("inf")):
        logger.info(
            f"Primary metric '{primary_metric_key}' for {model_type_name} models yielded score of {best_score}. Trying secondary metric '{secondary_metric_key}'."
        )
        best_score_secondary: float = -float("inf")
        best_name_secondary: Optional[str] = None
        for name, metrics in metrics_dict.items():
            score_sec = metrics.get(secondary_metric_key, -float("inf"))
            if score_sec > best_score_secondary:
                best_score_secondary = score_sec
                best_name_secondary = name

        if (
            best_name_secondary is not None and best_score_secondary > best_score
        ):  # Check if secondary is better or primary was nothing
            best_name = best_name_secondary
            best_score = best_score_secondary
            used_key = secondary_metric_key

    if best_name is None or best_score == -float("inf"):
        logger.warning(
            f"Could not determine best {model_type_name} model using keys '{primary_metric_key}' or '{secondary_metric_key}'. All scores might be -infinity or keys not found."
        )
        return None

    logger.info(
        f"Best {model_type_name} Model (by '{used_key}'): {best_name} (Score: {best_score:.4f})"
    )
    return best_name


def _save_artifact(obj: Any, filename: str, directory: Path = MODELS_DIR) -> None:
    """Helper to save an artifact using joblib, creating directory if needed."""
    if obj is None:
        logger.warning(f"Attempted to save a None object for '{filename}'. Skipping.")
        return

    file_path = directory / filename
    try:
        os.makedirs(directory, exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Saved artifact '{filename}' to {file_path}")
    except Exception as e:
        logger.error(f"Could not save artifact '{filename}' to {file_path}: {e}", exc_info=True)


def _load_artifact(filename: str, directory: Path = MODELS_DIR) -> Optional[Any]:
    """Helper to load an artifact using joblib."""
    file_path = directory / filename
    if file_path.exists():
        try:
            obj = joblib.load(file_path)
            logger.info(f"Successfully loaded artifact '{filename}' from {file_path}.")
            return obj
        except Exception as e:
            logger.error(
                f"Error loading artifact '{filename}' from {file_path}: {e}", exc_info=True
            )
            return None
    else:
        logger.warning(f"Artifact '{filename}' not found at {file_path}.")
        return None


# --- Core Pipeline Functions ---


def run_training_and_analysis(file_path_str: str) -> None:
    """
    Runs the full training and analysis pipeline.
    Updates global variables: `fitted_models_global`, `model_metrics_global`,
    `best_model_name_global`, `best_ensemble_model_name_global`, and `X_global`.
    """
    logger.info("--- Starting HIV Modeling Training and Analysis Pipeline ---")
    global \
        fitted_models_global, \
        model_metrics_global, \
        best_model_name_global, \
        best_ensemble_model_name_global, \
        X_global

    file_path: Path = PROCESSED_DATA_DIR / file_path_str

    # 1. Load Data
    logger.info(f"Step 1: Loading data from {file_path}...")
    df_weekly: Optional[pd.DataFrame] = load_data(file_path)
    if not _validate_dataframe_for_training(df_weekly, "df_weekly (loaded data)"):
        return
    assert df_weekly is not None  # For type checker
    logger.info("Data loading complete.")

    # 2. Prepare data for modeling
    logger.info("\nStep 2: Preparing data for modeling...")
    n_modeling_splits: int = 5
    if len(df_weekly) < n_modeling_splits * 2:
        logger.warning(
            f"Data length ({len(df_weekly)}) is very short for {n_modeling_splits} splits. CV might be unreliable."
        )

    X: np.ndarray
    y: np.ndarray
    cv_splits: List[Tuple[np.ndarray, np.ndarray]]
    tscv_object: CVObject
    try:
        X, y, cv_splits, tscv_object = prepare_data_for_modeling(
            df_weekly, n_splits=n_modeling_splits
        )
        X_global = X  # Store training X time indices globally
    except ValueError as e:
        logger.error(f"Error preparing data for modeling: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error during data preparation: {e}", exc_info=True)
        return
    logger.info("Data preparation complete.")

    X_train, y_train, X_test, y_test, final_train_idx, final_test_idx = (
        _prepare_train_test_split_from_cv(X, y, cv_splits)
    )

    # 3. Fit individual growth models
    logger.info("\nStep 3: Fitting individual growth models...")
    current_fitted_models, current_model_metrics, cv_fitted_models = fit_growth_models(
        X, y, cv_splits
    )
    fitted_models_global.clear()
    fitted_models_global.update(current_fitted_models)
    model_metrics_global.clear()
    model_metrics_global.update(current_model_metrics)
    if not fitted_models_global:
        logger.error("No individual models fitted. Exiting.")
        return
    logger.info(f"Individual model fitting complete. Models: {list(fitted_models_global.keys())}")

    # --- New: Output for Standalone Growth Model Results (Request 2) ---
    logger.info("\nStep 3.1: Generating reports and diagnostics for standalone growth models...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure reports directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)  # Ensure figures directory exists

    # --- Diagnostic plots for each standalone growth model ---
    logger.info("Generating diagnostic plots for individual growth models...")
    for model_name, model_details in fitted_models_global.items():
        if model_details and "function" in model_details and "parameters" in model_details:
            # Use full dataset (X, y) for these diagnostic plots for overall model assessment
            # Alternatively, could use final_test_idx if focusing on test set performance.
            # For overall model assumption checks, full data is often preferred.
            y_pred_model_full = model_details["function"](X.ravel(), *model_details["parameters"])
            residuals_model_full = y.ravel() - y_pred_model_full

            plot_residuals(
                y.ravel(),  # y_true
                y_pred_model_full,  # y_pred
                model_name,
                filename_suffix="_full_data",  # Added suffix for clarity
                filename=f"residuals_{model_name}_full_data.png",
            )
            plot_residuals_histogram(
                residuals_model_full,  # Pass residuals directly
                model_name,
                filename_suffix="_full_data",
                filename=f"residuals_histogram_{model_name}_full_data.png",
            )
            plot_qq(
                residuals_model_full,
                model_name,
                filename_suffix="_full_data",
                filename=f"qq_plot_{model_name}_full_data.png",
            )
            logger.info(
                f"Diagnostic plots for {model_name} (full data) generated in {FIGURES_DIR}."
            )
            logger.info(f"  Please check these plots for {model_name} to assess assumptions like:")
            logger.info(
                "    - Residual Plot: Random scatter around zero (homoscedasticity, linearity)."
            )
            logger.info(
                "    - Q-Q Plot: Points close to the diagonal line (normality of residuals)."
            )
            logger.info(
                "    - Histogram of Residuals: Bell-shaped curve (normality of residuals)."
            )
        else:
            logger.warning(
                f"Could not generate diagnostic plots for {model_name} due to missing details."
            )
    logger.info("Diagnostic plots for individual growth models complete.")
    # --- End of diagnostic plots section ---

    # 3.1.1. Model Coefficient Table
    coeff_table_df = generate_coefficient_table(fitted_models_global, model_metrics_global)
    coeff_table_path = REPORTS_DIR / "standalone_model_coefficients.csv"
    coeff_table_df.to_csv(coeff_table_path, index=False)
    logger.info(f"Standalone model coefficient table saved to {coeff_table_path}")
    print("\n--- Standalone Model Coefficients ---")
    print(coeff_table_df.to_string())

    # 3.1.2. Model Performance Metrics Table
    standalone_metrics_df = generate_standalone_metrics_table(model_metrics_global)
    standalone_metrics_path = REPORTS_DIR / "standalone_model_metrics.csv"
    standalone_metrics_df.to_csv(standalone_metrics_path, index=False)
    logger.info(f"Standalone model performance metrics table saved to {standalone_metrics_path}")
    print("\n--- Standalone Model Performance Metrics ---")
    print(standalone_metrics_df.to_string())

    # 3.1.3. Observed Vs. Predicted Charts (Individual)
    logger.info("Generating individual observed vs. predicted plots for standalone models...")
    for model_name, model_details in fitted_models_global.items():
        if model_details and "function" in model_details and "parameters" in model_details:
            visualize_single_model_fit(
                df=df_weekly,
                X=X,
                y=y,
                model_name=model_name,
                model_details=model_details,
                final_train_idx=final_train_idx,
                final_test_idx=final_test_idx,
                y_train=y_train,
                y_test=y_test,
                filename=f"observed_vs_predicted_{model_name}.png",
            )
    logger.info("Individual observed vs. predicted plots generated.")
    # --- End of New Section for Request 2 ---

    # 4. Visualize individual model fits (Combined Plot)
    logger.info("\nStep 4: Visualizing combined individual model fits...")
    visualize_individual_models(
        df_weekly,
        X,
        final_train_idx,
        final_test_idx,
        y_train,
        y_test,
        fitted_models_global,
        filename="individual_model_fits.png",
    )
    logger.info("Combined individual model fit visualization complete.")

    # 5. Build ensemble models
    logger.info("\nStep 5: Building ensemble models...")
    # Pass final fitted growth models and their metrics for building average-based ensembles
    ensemble_models_dict, ensemble_metrics_dict = build_ensemble_models_with_cv(
        X, y, cv_fitted_models, cv_splits
    )
    if not ensemble_models_dict:
        logger.warning("No ensemble models built.")
        ensemble_metrics_dict = {}
    logger.info(f"Ensemble model building complete. Models: {list(ensemble_models_dict.keys())}")

    # --- New: Output for Ensemble Model Results (Request 3) ---
    logger.info("\nStep 5.1: Generating reports for ensemble models...")
    if ensemble_models_dict and fitted_models_global:  # Ensure models exist
        # Prepare X_time_idx as a Series for reporting functions (assuming X is time_idx)
        # Use the full X, y for these reports, not just train/test split from last CV
        X_time_idx_series = pd.Series(X.ravel(), name="Day")
        y_actual_series = pd.Series(y.ravel(), name="Actual")
        growth_model_names_list = list(fitted_models_global.keys())

        # 5.1.1. Input Table (Combined Predictions for Each Ensemble Technique)
        logger.info("Generating ensemble input tables...")
        ensemble_input_tables_dict = generate_ensemble_input_tables(
            X_time_idx=X_time_idx_series,
            y_actual=y_actual_series,  # Not strictly used in current table, but good to pass
            fitted_growth_models=fitted_models_global,
            ensemble_models_dict=ensemble_models_dict,
            growth_model_names=growth_model_names_list,
        )
        for ens_name, ens_df in ensemble_input_tables_dict.items():
            ens_input_table_path = (
                REPORTS_DIR / f"ensemble_input_table_{ens_name.replace(' ', '_')}.csv"
            )
            ens_df.to_csv(ens_input_table_path, index=False)
            logger.info(f"Ensemble input table for {ens_name} saved to {ens_input_table_path}")
            # print(f"\n--- Ensemble Input Table: {ens_name} ---") # Optional: print to console
            # print(ens_df.head().to_string())

        # 5.1.2. Full Prediction Table
        logger.info("Generating full predictions table...")
        full_preds_df = generate_full_predictions_table(
            X_time_idx=X_time_idx_series,
            y_actual=y_actual_series,
            fitted_growth_models=fitted_models_global,
            ensemble_models_dict=ensemble_models_dict,
            growth_model_names=growth_model_names_list,
        )
        full_preds_table_path = REPORTS_DIR / "full_predictions_table.csv"
        full_preds_df.to_csv(full_preds_table_path, index=False)
        logger.info(f"Full predictions table saved to {full_preds_table_path}")
        # print("\n--- Full Predictions Table (First 5 rows) ---") # Optional: print to console
        # print(full_preds_df.head().to_string())

        # 5.1.3. Summary Table of Full Predictions
        if not full_preds_df.empty:
            logger.info("Generating predictions summary table...")
            preds_summary_df = generate_predictions_summary_table(full_preds_df)
            preds_summary_table_path = REPORTS_DIR / "predictions_summary_table.csv"
            preds_summary_df.to_csv(preds_summary_table_path, index=False)
            logger.info(f"Predictions summary table saved to {preds_summary_table_path}")
            print("\n--- Predictions Summary Table ---")
            print(preds_summary_df.to_string())
        else:
            logger.warning(
                "Full predictions DataFrame is empty, skipping summary table generation."
            )

        # --- New: Weighted Average Specific Reports (Request 3, continued) ---
        # 5.1.4. Weighted Average Comparison Table
        if ensemble_metrics_dict:
            logger.info("Generating weighted average comparison table...")
            wa_comparison_df = generate_weighted_average_comparison_table(ensemble_metrics_dict)
            wa_comparison_path = REPORTS_DIR / "weighted_average_comparison.csv"
            wa_comparison_df.to_csv(wa_comparison_path, index=False)
            logger.info(f"Weighted average comparison table saved to {wa_comparison_path}")
            print("\n--- Weighted Average Performance Comparison ---")
            print(wa_comparison_df.to_string())

            # Visualize the weighted average comparison
            if not wa_comparison_df.empty:
                visualize_weighted_average_metrics_comparison(
                    wa_comparison_df, filename="weighted_average_metrics_comparison.png"
                )
                logger.info(
                    f"Weighted average metrics comparison chart saved to {FIGURES_DIR / 'weighted_average_metrics_comparison.png'}"
                )
            else:
                logger.warning(
                    "Weighted average comparison DataFrame is empty, skipping chart generation."
                )

        # 5.1.5. Weighted Parameters Table
        if model_metrics_global and ensemble_models_dict:
            logger.info("Generating weighted parameters table...")
            # Pass model_metrics_global for growth rates, ensemble_models_dict for actual weights
            weighted_params_df = generate_weighted_parameters_table(
                model_metrics_global, ensemble_models_dict
            )
            weighted_params_path = REPORTS_DIR / "weighted_parameters_table.csv"
            weighted_params_df.to_csv(weighted_params_path, index=False)
            logger.info(f"Weighted parameters table saved to {weighted_params_path}")
            print("\n--- Weighted Parameters Table (Growth Models) ---")
            print(weighted_params_df.to_string())
        # --- End of Weighted Average Specific Reports ---

        # --- New: Feature Importance Reports (Request 4, continued) ---
        # 5.1.6. Feature Importance Table and Plots for RF & GB
        if ensemble_models_dict:
            logger.info("Generating feature importance table for RF & GB...")
            feature_importance_df = generate_feature_importance_table(ensemble_models_dict)
            if not feature_importance_df.empty:
                fi_table_path = REPORTS_DIR / "feature_importances_RF_GB.csv"
                feature_importance_df.to_csv(fi_table_path, index=False)
                logger.info(f"Feature importance table saved to {fi_table_path}")
                print("\n--- Feature Importances (RF & GB) ---")
                print(feature_importance_df.to_string())

                logger.info("Generating feature importance plots for RF & GB...")
                for model_key_for_fi in ["Random Forest", "Gradient Boosting"]:
                    if model_key_for_fi in ensemble_models_dict and ensemble_models_dict[
                        model_key_for_fi
                    ].get("feature_names"):
                        plot_feature_importances(
                            ensemble_models_dict,  # Pass the full dict
                            model_name_key=model_key_for_fi,  # Specify which model to plot from the dict
                            model_display_name=model_key_for_fi,  # For plot title
                            filename=f"feature_importances_{model_key_for_fi.replace(' ', '_')}.png",
                            top_n=15,
                        )
                        logger.info(f"Feature importance plot for {model_key_for_fi} saved.")
                    else:
                        logger.info(
                            f"Skipping feature importance plot for {model_key_for_fi} - model or features not found."
                        )
            else:
                logger.warning(
                    "Feature importance DataFrame is empty. Skipping table save and plots."
                )
        # --- End of Feature Importance Reports ---
    else:
        logger.warning(
            "Skipping ensemble model reports generation: No ensemble or fitted growth models available."
        )
    # --- End of New Section for Request 3 ---

    # 6. Identify best models (updates globals)
    logger.info("\nStep 6: Identifying best models...")
    best_model_name_global = _determine_best_model(model_metrics_global, "Individual Growth")
    best_ensemble_model_name_global = _determine_best_model(
        ensemble_metrics_dict, "Ensemble", primary_metric_key="test_r2", secondary_metric_key="r2"
    )

    # Visualization and Plotting Steps (7-11)
    _perform_post_modeling_visualizations(
        df_weekly,
        X,
        y,
        cv_splits,
        final_train_idx,
        final_test_idx,
        y_train,
        y_test,
        fitted_models_global,
        model_metrics_global,
        best_model_name_global,
        ensemble_models_dict,
        ensemble_metrics_dict,
        best_ensemble_model_name_global,
        generate_bootstrap_predictions,
    )
    logger.info("\n--- Analysis and training complete. ---")

    # --- Interpretation and Summary Guidance ---
    logger.info("\n--- Interpretation and Summary ---")
    logger.info("All requested tables and charts have been generated and saved.")
    logger.info(f"Reports (CSVs) are in: {REPORTS_DIR}")
    logger.info(f"Figures (PNGs) are in: {FIGURES_DIR}")
    logger.info("\nTo interpret the results, please consider the following:")
    logger.info(
        f"1. Best Overall Model: Refer to '{REPORTS_DIR / 'overall_evaluation_metrics.csv'}' "
        f"and '{FIGURES_DIR / 'model_metrics_comparison.png'}'"
    )
    logger.info("   Look for models with low RMSE/MAE and high R².")
    logger.info(
        "2. Ensemble vs. Individual Models: Compare metrics of 'Ensemble' type models "
        "against 'Individual' type models in the same files mentioned above."
    )
    logger.info(
        f"3. Ensemble Technique Stability: Analyze metrics in '{REPORTS_DIR / 'overall_evaluation_metrics.csv'}'. "
        "Lower variance in performance across different datasets (if applicable through CV results not directly shown here) "
        "or consistent high performance suggests stability. Also, review residual plots for ensemble models "
        f"(e.g., '{FIGURES_DIR / 'residuals_plot_Random_Forest.png'}') for patterns."
    )
    logger.info(
        f"4. Prediction Accuracy Over Time: Examine plots like '{FIGURES_DIR / 'best_model_vs_ensemble.png'}' (validation plot), "
        f"individual model fit plots (e.g., '{FIGURES_DIR / 'observed_vs_predicted_Logistic.png'}'), "
        f"and '{FIGURES_DIR / 'ensemble_comparison.png'}'. Look for divergence between predicted and actual values over time."
    )
    logger.info(
        f"5. Growth Model Assumptions: Check the diagnostic plots for individual growth models "
        f"(e.g., '{FIGURES_DIR / 'residuals_Logistic_full_data.png'}', "
        f"'{FIGURES_DIR / 'qq_plot_Logistic_full_data.png'}') to assess if model assumptions are met."
    )
    logger.info(
        "This automated pipeline provides the data and visualizations; detailed narrative interpretation requires domain expertise."
    )
    logger.info("--- End of Interpretation and Summary Guidance ---")


def _perform_post_modeling_visualizations(
    df_weekly: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    final_train_idx: np.ndarray,
    final_test_idx: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    fitted_models: Dict[str, ModelType],
    model_metrics: Dict[str, Dict[str, float]],
    best_model_name: Optional[str],
    ensemble_models_dict: Dict[str, ModelType],
    ensemble_metrics_dict: Dict[str, Dict[str, float]],
    best_ensemble_name: Optional[str],
    bootstrap_func: Callable,
) -> None:
    """Handles visualization steps 7-11 from the original main block."""

    # 7. Visualize ensemble comparisons
    logger.info("\nStep 7: Visualizing ensemble comparisons...")
    if ensemble_models_dict and best_model_name and best_model_name in fitted_models:
        visualize_ensemble_comparison(
            df_weekly,
            X,
            y,
            cv_splits,
            fitted_models,
            ensemble_models_dict,
            best_model_name,
            filename="ensemble_comparison.png",
        )
        logger.info("Ensemble comparison visualization complete.")
    else:
        logger.warning(
            "Skipping ensemble comparison: missing data (best_model_name, ensemble_models_dict, or model not in fitted_models)."
        )

    # 8. Compare model metrics
    logger.info("\nStep 8: Visualizing metrics comparison...")
    # visualize_metrics_comparison now returns fig, metrics_df
    _, all_metrics_df = visualize_metrics_comparison(
        model_metrics, ensemble_metrics_dict, filename="model_metrics_comparison.png"
    )
    logger.info("Metrics comparison visualization complete.")

    # Save the combined metrics table (Request 4)
    if all_metrics_df is not None and not all_metrics_df.empty:
        all_metrics_table_path = REPORTS_DIR / "overall_evaluation_metrics.csv"
        # Select and rename columns to match request: Model, RMSE, MAE, R²
        # The df from visualize_metrics_comparison has 'Model', 'Type', 'RMSE', 'R²', 'MAE'
        # We can keep 'Type' as it's informative.
        report_cols = ["Model", "Type", "RMSE", "R²", "MAE"]
        cols_to_save = [col for col in report_cols if col in all_metrics_df.columns]
        all_metrics_df_to_save = all_metrics_df[cols_to_save]

        all_metrics_df_to_save.to_csv(all_metrics_table_path, index=False)
        logger.info(f"Overall evaluation metrics table saved to {all_metrics_table_path}")
        print("\n--- Overall Evaluation Metrics ---")
        print(all_metrics_df_to_save.to_string())
    else:
        logger.warning("Metrics DataFrame for overall evaluation is empty or None. Skipping save.")

    best_ind_model_obj = fitted_models.get(str(best_model_name)) if best_model_name else None
    best_ens_model_obj = (
        ensemble_models_dict.get(str(best_ensemble_name)) if best_ensemble_name else None
    )

    # 9. Create validation plot
    logger.info("\nStep 9: Creating validation plot...")
    if best_ind_model_obj and best_ens_model_obj and best_model_name and best_ensemble_name:
        create_validation_plot(
            df_weekly,
            X,
            y,
            best_ind_model_obj,
            best_ens_model_obj,
            str(best_ensemble_name),
            str(best_model_name),
            filename="best_model_vs_ensemble.png",
        )
        logger.info("Validation plot creation complete.")
    else:
        logger.warning("Skipping validation plot: missing best model objects or names.")

    # 10. Diagnostic Plots for All Ensemble Models
    logger.info("\nStep 10: Generating diagnostic plots for all ensemble models...")
    X_diag_ensemble, y_diag_ensemble = (
        (X[final_test_idx], y[final_test_idx])
        if final_test_idx.size > 0 and y[final_test_idx].size > 0
        else (X, y)  # Use full data if test set is empty
    )

    if X_diag_ensemble.size > 0 and y_diag_ensemble.size > 0 and ensemble_models_dict:
        for ens_model_name, ens_model_obj in ensemble_models_dict.items():
            if ens_model_obj and "predict" in ens_model_obj:
                try:
                    logger.info(f"Generating diagnostic plots for ensemble: {ens_model_name}...")
                    y_pred_ens = ens_model_obj["predict"](X_diag_ensemble)
                    residuals_ens = y_diag_ensemble - y_pred_ens

                    # Sanitize filename
                    safe_model_name = (
                        ens_model_name.replace(" ", "_")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("/", "")
                    )

                    plot_residuals(
                        y_diag_ensemble,
                        y_pred_ens,
                        ens_model_name,
                        filename=f"residuals_plot_{safe_model_name}.png",
                    )
                    plot_residuals_histogram(
                        residuals_ens,  # Pass residuals directly
                        ens_model_name,
                        filename=f"residuals_histogram_{safe_model_name}.png",
                    )
                    plot_qq(
                        residuals_ens,
                        ens_model_name,
                        filename=f"qq_plot_{safe_model_name}.png",
                    )
                    logger.info(f"Diagnostic plots for {ens_model_name} complete.")

                    # Feature importance plots are handled in Step 5.1.6 of run_training_and_analysis
                    # as they are tied to specific model types (RF, GB) and use their specific attributes.
                    # No need to replicate that logic here.

                except Exception as e:
                    logger.error(
                        f"Could not generate diagnostic plots for {ens_model_name}: {e}",
                        exc_info=True,
                    )
            else:
                logger.warning(
                    f"Predict function or model object not found for {ens_model_name}, skipping its diagnostic plots."
                )
    else:
        logger.warning(
            "Skipping ensemble diagnostic plots: No diagnostic data or no ensemble models available."
        )

    # 11. Forecast future trends
    logger.info("\nStep 11: Forecasting future trends...")
    if best_ind_model_obj and best_ens_model_obj and best_model_name and best_ensemble_name:
        forecast_future_trends(
            df_weekly,
            X,
            y,
            best_ind_model_obj,
            best_ens_model_obj,
            str(best_ensemble_name),
            fitted_models,
            str(best_model_name),
            bootstrap_func,
            filename="future_forecast.png",
        )
        logger.info("Future trends forecasting complete.")
    else:
        logger.warning("Skipping future trends forecast: missing best model objects or names.")


def load_predict_models_and_scaler() -> Tuple[
    Optional[ModelType], Optional[ModelType], Optional[ScalerType]
]:
    """Loads RF, GB models and scaler, expecting specific filenames."""
    logger.info(
        f"Loading primary ML models (rf_model.pkl, gb_model.pkl) and scaler (feature_scaler.pkl) from {MODELS_DIR}..."
    )
    rf_model = _load_artifact("rf_model.pkl")
    gb_model = _load_artifact("gb_model.pkl")
    scaler = _load_artifact("feature_scaler.pkl")

    if scaler is None:
        logger.error("CRITICAL: Feature scaler could not be loaded. Prediction will likely fail.")
    if rf_model is None and gb_model is None:
        logger.warning("Neither RF nor GB model loaded.")
    return rf_model, gb_model, scaler


def generate_forecasts(
    df_predict: pd.DataFrame,
    ml_model: ModelType,
    scaler: ScalerType,
    fitted_growth_models: Dict[str, ModelType],
    X_train_context: Optional[np.ndarray],
) -> Optional[pd.DataFrame]:
    """Generates predictions using a trained ML model, scaler, and growth model context."""
    if "time_idx" not in df_predict.columns:
        logger.error("Prediction DataFrame must contain 'time_idx'.")
        return None
    if ml_model is None:
        logger.error("ML model not provided.")
        return None
    if scaler is None:
        logger.error("Scaler not provided.")
        return None

    X_pred_time_idx: np.ndarray = np.array(df_predict["time_idx"].values).reshape(-1, 1)
    logger.info("Generating ML features for forecasting...")

    context_for_growth_features = (
        X_train_context if X_train_context is not None else X_pred_time_idx
    )
    if X_train_context is None and fitted_growth_models:
        logger.warning(
            "`X_train_context` is None, but growth models provided. Using prediction time indices for growth feature context."
        )

    try:
        ml_features = create_ml_features(
            X=X_pred_time_idx,
            fitted_models=fitted_growth_models,
            X_full=context_for_growth_features,
        )
    except Exception as e:
        logger.error(f"Error during ML feature creation: {e}", exc_info=True)
        return None

    if isinstance(ml_features, pd.DataFrame) and ml_features.empty:
        logger.error("Feature creation returned empty DataFrame.")
        return None

    try:
        ml_features_scaled: np.ndarray = scaler.transform(ml_features)
    except Exception as e:
        logger.error(
            f"Error scaling features: {e}. Columns: {ml_features.columns.tolist() if isinstance(ml_features, pd.DataFrame) else 'N/A'}",
            exc_info=True,
        )
        return None

    logger.info(f"Generating predictions with {type(ml_model).__name__}...")
    try:
        predictions: np.ndarray = ml_model.predict(ml_features_scaled)
    except Exception as e:
        logger.error(f"Error during model prediction: {e}", exc_info=True)
        return None

    forecast_output_df = df_predict.copy()
    forecast_output_df["predicted_cumulative"] = predictions
    return forecast_output_df


# --- Main Pipeline Orchestration ---


def main_pipeline(
    mode: str = "train_and_analyze", data_file: str = "cleaned_enrollments.csv"
) -> None:
    """Main entry point for training & analysis, or prediction."""
    global fitted_models_global, X_global  # Referenced for saving/loading context

    if mode == "train_and_analyze":
        run_training_and_analysis(file_path_str=data_file)

        logger.info("Post-training: Saving key artifacts for standalone prediction...")
        if best_model_name_global and best_model_name_global in fitted_models_global:
            _save_artifact(
                fitted_models_global[best_model_name_global],
                f"best_individual_growth_{best_model_name_global}.pkl",
            )

        if fitted_models_global:
            _save_artifact(fitted_models_global, "all_fitted_growth_models.pkl")
        if X_global is not None:
            _save_artifact(X_global, "X_train_context_for_growth_models.pkl")
        logger.info(
            "Reminder: `build_ensemble_models` should save 'feature_scaler.pkl' and ML models (e.g., 'rf_model.pkl') for predict mode."
        )

    elif mode == "predict":
        logger.info("--- Starting Prediction Pipeline ---")
        data_path = PROCESSED_DATA_DIR / data_file
        df_to_predict = load_data(data_path)

        if df_to_predict is None or df_to_predict.empty or "time_idx" not in df_to_predict.columns:
            logger.error(
                f"Invalid data for prediction from {data_path}. Must contain 'time_idx'. Exiting."
            )
            return

        rf_model, gb_model, loaded_scaler = load_predict_models_and_scaler()
        if loaded_scaler is None:
            logger.error("Scaler is critical and missing. Prediction aborted.")
            return

        ml_model_to_use = rf_model if rf_model else gb_model
        if not ml_model_to_use:
            logger.error("No suitable ML model (RF or GB) loaded. Prediction aborted.")
            return

        logger.info(f"Using {type(ml_model_to_use).__name__} for prediction.")

        # Load growth models and training X context for feature engineering
        predict_fitted_growth_models = _load_artifact("all_fitted_growth_models.pkl") or {}
        if not predict_fitted_growth_models and fitted_models_global:  # Fallback to global
            logger.info(
                "Using growth models from current session's global state (training likely run previously)."
            )
            predict_fitted_growth_models = fitted_models_global

        predict_X_train_context = _load_artifact("X_train_context_for_growth_models.pkl")
        if predict_X_train_context is None and X_global is not None:  # Fallback to global
            logger.info("Using X_train_context from current session's global state.")
            predict_X_train_context = X_global

        if not predict_fitted_growth_models:
            logger.warning(
                "No growth models loaded or found in globals; features from them will be absent."
            )
        if predict_X_train_context is None and predict_fitted_growth_models:
            logger.warning(
                "Growth models loaded, but X_train_context is missing; feature generation context might be suboptimal."
            )

        forecast_df = generate_forecasts(
            df_to_predict,
            ml_model_to_use,
            loaded_scaler,
            predict_fitted_growth_models,
            predict_X_train_context,
        )

        if forecast_df is not None:
            base_name = data_file.split(".")[0] if "." in data_file else data_file
            output_filename = f"forecast_results_{base_name}.csv"
            _save_artifact(
                forecast_df, output_filename, directory=PROCESSED_DATA_DIR
            )  # Save forecasts
        else:
            logger.error("Forecast generation failed.")
        logger.info("--- Prediction Pipeline Complete ---")
    else:
        logger.error(f"Invalid mode: {mode}. Choose 'train_and_analyze' or 'predict'.")


if __name__ == "__main__":
    # Ensure necessary directories exist
    for dir_path in [PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # --- Run Training and Analysis ---
    logger.info("--- Main Execution: Starting Training and Analysis Phase ---")
    # Assumes "cleaned_enrollments.csv" exists or can be generated by `load_data` if it handles that.
    # If `load_data` only loads, the file must be present.
    main_pipeline(mode="train_and_analyze", data_file="cleaned_enrollments.csv")
    logger.info("--- Main Execution: Training and Analysis Phase Complete ---")
    main_pipeline(mode="predict")
