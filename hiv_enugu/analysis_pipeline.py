"""
This script consolidates the functionality of run_analysis.py and train.py
for HIV enrollment forecasting in Enugu.
It provides a pipeline for training models, analyzing results, and generating predictions.
"""
import pandas as pd
import numpy as np
import warnings
import os
import joblib
from loguru import logger
from typing import List, Dict, Tuple, Optional, Any, Callable

# Custom module imports
from hiv_enugu.data_processing import load_data
from hiv_enugu.modeling.data_prep import prepare_data_for_modeling
from hiv_enugu.modeling.fit import fit_growth_models
from hiv_enugu.modeling.ensemble import build_ensemble_models
from hiv_enugu.utils import generate_bootstrap_predictions # generate_bootstrap_predictions type: Callable
from hiv_enugu.modeling.features import create_ml_features

from hiv_enugu.plotting.evaluation import (
    visualize_individual_models,
    visualize_ensemble_comparison,
    visualize_metrics_comparison,
    create_validation_plot,
    forecast_future_trends
)
from hiv_enugu.plotting.diagnostics import plot_residuals, plot_residuals_histogram
from hiv_enugu.config import PROCESSED_DATA_DIR, FIGURES_DIR, MODELS_DIR

warnings.filterwarnings('ignore')

# Type Aliases
ModelType = Any  # General placeholder for any model type (e.g., scikit-learn estimator)
ScalerType = Any # General placeholder for a scaler (e.g., StandardScaler)
CVObject = Any   # Placeholder for TimeSeriesSplit object or similar

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

def _validate_dataframe_for_training(df: Optional[pd.DataFrame], df_name: str = "DataFrame") -> bool:
    """Validates if the DataFrame is suitable for training."""
    if df is None or df.empty:
        logger.error(f"{df_name} is None or empty. Cannot proceed with training.")
        return False
    if not isinstance(df, pd.DataFrame):
        logger.error(f"{df_name} is not a Pandas DataFrame. Cannot proceed.")
        return False
    required_cols = ['time_idx', 'cumulative']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Error: '{df_name}' must contain {required_cols} columns for training. Found: {df.columns.tolist()}.")
        return False
    return True


def _prepare_train_test_split_from_cv(
    X: np.ndarray, y: np.ndarray, cv_splits: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts training/testing sets from the last CV split, handling empty or out-of-bounds cases."""
    final_train_idx, final_test_idx = (cv_splits[-1] if cv_splits else (np.array([]), np.array([])))

    def get_slice(data_array: np.ndarray, indices: np.ndarray, default_array: np.ndarray) -> np.ndarray:
        if indices.size > 0 and data_array.size > 0 and indices.max() < data_array.shape[0]:
            return data_array[indices]
        return default_array

    X_train = get_slice(X, final_train_idx, X) # Default to full X if no valid train indices
    y_train = get_slice(y, final_train_idx, y) # Default to full y
    X_test = get_slice(X, final_test_idx, np.array([])) # Default to empty if no valid test indices
    y_test = get_slice(y, final_test_idx, np.array([]))

    return X_train, y_train, X_test, y_test, final_train_idx, final_test_idx


def _determine_best_model(
    metrics_dict: Dict[str, Dict[str, float]],
    model_type_name: str,
    primary_metric_key: str = 'test_r2',
    secondary_metric_key: Optional[str] = 'r2'
) -> Optional[str]:
    """Determines the best model from metrics, trying primary then secondary keys."""
    if not metrics_dict:
        logger.warning(f"Cannot determine best {model_type_name} model: Metrics dictionary is empty.")
        return None

    best_name: Optional[str] = None
    best_score: float = -float('inf') # Initialize with negative infinity
    used_key: str = primary_metric_key

    # Try primary key
    for name, metrics in metrics_dict.items():
        score = metrics.get(primary_metric_key, -float('inf'))
        if score > best_score:
            best_score = score
            best_name = name

    # If primary key yielded no valid model or score, try secondary key
    if secondary_metric_key and (best_name is None or best_score == -float('inf')):
        logger.info(f"Primary metric '{primary_metric_key}' for {model_type_name} models yielded score of {best_score}. Trying secondary metric '{secondary_metric_key}'.")
        best_score_secondary: float = -float('inf')
        best_name_secondary: Optional[str] = None
        for name, metrics in metrics_dict.items():
            score_sec = metrics.get(secondary_metric_key, -float('inf'))
            if score_sec > best_score_secondary:
                best_score_secondary = score_sec
                best_name_secondary = name

        if best_name_secondary is not None and best_score_secondary > best_score: # Check if secondary is better or primary was nothing
            best_name = best_name_secondary
            best_score = best_score_secondary
            used_key = secondary_metric_key

    if best_name is None or best_score == -float('inf'):
        logger.warning(f"Could not determine best {model_type_name} model using keys '{primary_metric_key}' or '{secondary_metric_key}'. All scores might be -infinity or keys not found.")
        return None

    logger.info(f"Best {model_type_name} Model (by '{used_key}'): {best_name} (Score: {best_score:.4f})")
    return best_name

def _save_artifact(obj: Any, filename: str, directory: pd.io.common.PathIsh = MODELS_DIR) -> None:
    """Helper to save an artifact using joblib, creating directory if needed."""
    if obj is None:
        logger.warning(f"Attempted to save a None object for '{filename}'. Skipping.")
        return
    try:
        os.makedirs(directory, exist_ok=True)
        file_path = directory / filename
        joblib.dump(obj, file_path)
        logger.info(f"Saved artifact '{filename}' to {file_path}")
    except Exception as e:
        logger.error(f"Could not save artifact '{filename}' to {file_path}: {e}", exc_info=True)

def _load_artifact(filename: str, directory: pd.io.common.PathIsh = MODELS_DIR) -> Optional[Any]:
    """Helper to load an artifact using joblib."""
    file_path = directory / filename
    if file_path.exists():
        try:
            obj = joblib.load(file_path)
            logger.info(f"Successfully loaded artifact '{filename}' from {file_path}.")
            return obj
        except Exception as e:
            logger.error(f"Error loading artifact '{filename}' from {file_path}: {e}", exc_info=True)
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
    global fitted_models_global, model_metrics_global, best_model_name_global, best_ensemble_model_name_global, X_global

    file_path: pd.io.common.PathIsh = PROCESSED_DATA_DIR / file_path_str

    # 1. Load Data
    logger.info(f"Step 1: Loading data from {file_path}...")
    df_weekly: Optional[pd.DataFrame] = load_data(file_path)
    if not _validate_dataframe_for_training(df_weekly, "df_weekly (loaded data)"):
        return
    assert df_weekly is not None # For type checker
    logger.info("Data loading complete.")

    # 2. Prepare data for modeling
    logger.info("\nStep 2: Preparing data for modeling...")
    n_modeling_splits: int = 5
    if len(df_weekly) < n_modeling_splits * 2:
        logger.warning(f"Data length ({len(df_weekly)}) is very short for {n_modeling_splits} splits. CV might be unreliable.")

    X: np.ndarray; y: np.ndarray; cv_splits: List[Tuple[np.ndarray, np.ndarray]]; tscv_object: CVObject
    try:
        X, y, cv_splits, tscv_object = prepare_data_for_modeling(df_weekly, n_splits=n_modeling_splits)
        X_global = X # Store training X time indices globally
    except ValueError as e:
        logger.error(f"Error preparing data for modeling: {e}"); return
    except Exception as e:
        logger.error(f"Unexpected error during data preparation: {e}", exc_info=True); return
    logger.info("Data preparation complete.")

    X_train, y_train, X_test, y_test, final_train_idx, final_test_idx = _prepare_train_test_split_from_cv(X, y, cv_splits)

    # 3. Fit individual growth models
    logger.info("\nStep 3: Fitting individual growth models...")
    current_fitted_models, current_model_metrics = fit_growth_models(X, y, cv_splits)
    fitted_models_global.clear(); fitted_models_global.update(current_fitted_models)
    model_metrics_global.clear(); model_metrics_global.update(current_model_metrics)
    if not fitted_models_global: logger.error("No individual models fitted. Exiting."); return
    logger.info(f"Individual model fitting complete. Models: {list(fitted_models_global.keys())}")

    # 4. Visualize individual model fits
    logger.info("\nStep 4: Visualizing individual model fits...")
    visualize_individual_models(df_weekly, X, final_train_idx, final_test_idx, y_train, y_test, fitted_models_global, filename='individual_model_fits.png')
    logger.info("Individual model fit visualization complete.")

    # 5. Build ensemble models
    logger.info("\nStep 5: Building ensemble models...")
    # This function is expected to save 'feature_scaler.pkl' and potentially ML models like 'rf_model.pkl'.
    ensemble_models_dict, ensemble_metrics_dict = build_ensemble_models(X, y, fitted_models_global, model_metrics_global, cv_splits)
    if not ensemble_models_dict: logger.warning("No ensemble models built."); ensemble_metrics_dict = {}
    logger.info(f"Ensemble model building complete. Models: {list(ensemble_models_dict.keys())}")

    # 6. Identify best models (updates globals)
    logger.info("\nStep 6: Identifying best models...")
    best_model_name_global = _determine_best_model(model_metrics_global, "Individual Growth")
    best_ensemble_model_name_global = _determine_best_model(ensemble_metrics_dict, "Ensemble", primary_metric_key='test_r2', secondary_metric_key='r2')

    # Visualization and Plotting Steps (7-11)
    _perform_post_modeling_visualizations(
        df_weekly, X, y, cv_splits, final_train_idx, final_test_idx, y_train, y_test,
        fitted_models_global, model_metrics_global, best_model_name_global,
        ensemble_models_dict, ensemble_metrics_dict, best_ensemble_model_name_global,
        generate_bootstrap_predictions
    )
    logger.info("\n--- Analysis and training complete. ---")

def _perform_post_modeling_visualizations(
    df_weekly: pd.DataFrame, X: np.ndarray, y: np.ndarray, cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    final_train_idx: np.ndarray, final_test_idx: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
    fitted_models: Dict[str, ModelType], model_metrics: Dict[str, Dict[str, float]], best_model_name: Optional[str],
    ensemble_models_dict: Dict[str, ModelType], ensemble_metrics_dict: Dict[str, Dict[str, float]], best_ensemble_name: Optional[str],
    bootstrap_func: Callable
) -> None:
    """Handles visualization steps 7-11 from the original main block."""

    # 7. Visualize ensemble comparisons
    logger.info("\nStep 7: Visualizing ensemble comparisons...")
    if ensemble_models_dict and best_model_name and best_model_name in fitted_models:
        visualize_ensemble_comparison(df_weekly, X, y, cv_splits, fitted_models, ensemble_models_dict, best_model_name, filename='ensemble_comparison.png')
        logger.info("Ensemble comparison visualization complete.")
    else:
        logger.warning("Skipping ensemble comparison: missing data (best_model_name, ensemble_models_dict, or model not in fitted_models).")

    # 8. Compare model metrics
    logger.info("\nStep 8: Visualizing metrics comparison...")
    visualize_metrics_comparison(model_metrics, ensemble_metrics_dict, filename='model_metrics_comparison.png')
    logger.info("Metrics comparison visualization complete.")

    best_ind_model_obj = fitted_models.get(str(best_model_name)) if best_model_name else None
    best_ens_model_obj = ensemble_models_dict.get(str(best_ensemble_name)) if best_ensemble_name else None

    # 9. Create validation plot
    logger.info("\nStep 9: Creating validation plot...")
    if best_ind_model_obj and best_ens_model_obj and best_model_name and best_ensemble_name:
        create_validation_plot(df_weekly, X, y, best_ind_model_obj, best_ens_model_obj, str(best_ensemble_name), str(best_model_name), filename='best_model_vs_ensemble.png')
        logger.info("Validation plot creation complete.")
    else:
        logger.warning("Skipping validation plot: missing best model objects or names.")

    # 10. Diagnostic Plots for Best Ensemble Model
    logger.info("\nStep 10: Generating diagnostic plots for best ensemble model...")
    if best_ens_model_obj and hasattr(best_ens_model_obj, 'predict') and callable(getattr(best_ens_model_obj, 'predict')) and best_ensemble_name:
        X_diag, y_diag = (X[final_test_idx], y[final_test_idx]) if final_test_idx.size > 0 and y[final_test_idx].size > 0 else (X, y)
        if X_diag.size > 0 and y_diag.size > 0:
            y_pred_ensemble = getattr(best_ens_model_obj, 'predict')(X_diag)
            plot_residuals(y_diag, y_pred_ensemble, str(best_ensemble_name), filename="residuals_plot_ensemble.png")
            plot_residuals_histogram(y_diag, y_pred_ensemble, str(best_ensemble_name), filename="residuals_histogram_ensemble.png")
            logger.info("Diagnostic plots complete.")
        else:
            logger.warning("Skipping diagnostic plots: No diagnostic data (X_diag/y_diag empty).")
    else:
        logger.warning("Skipping diagnostic plots: Best ensemble model/name not available or no 'predict' method.")

    # 11. Forecast future trends
    logger.info("\nStep 11: Forecasting future trends...")
    if best_ind_model_obj and best_ens_model_obj and best_model_name and best_ensemble_name:
        forecast_future_trends(df_weekly, X, y, best_ind_model_obj, best_ens_model_obj, str(best_ensemble_name), fitted_models, str(best_model_name), bootstrap_func, filename='future_forecast.png')
        logger.info("Future trends forecasting complete.")
    else:
        logger.warning("Skipping future trends forecast: missing best model objects or names.")


def load_predict_models_and_scaler() -> Tuple[Optional[ModelType], Optional[ModelType], Optional[ScalerType]]:
    """Loads RF, GB models and scaler, expecting specific filenames."""
    logger.info(f"Loading primary ML models (rf_model.pkl, gb_model.pkl) and scaler (feature_scaler.pkl) from {MODELS_DIR}...")
    rf_model = _load_artifact("rf_model.pkl")
    gb_model = _load_artifact("gb_model.pkl")
    scaler = _load_artifact("feature_scaler.pkl")

    if scaler is None: logger.error("CRITICAL: Feature scaler could not be loaded. Prediction will likely fail.")
    if rf_model is None and gb_model is None: logger.warning("Neither RF nor GB model loaded.")
    return rf_model, gb_model, scaler


def generate_forecasts(
    df_predict: pd.DataFrame,
    ml_model: ModelType,
    scaler: ScalerType,
    fitted_growth_models: Dict[str, ModelType],
    X_train_context: Optional[np.ndarray]
) -> Optional[pd.DataFrame]:
    """Generates predictions using a trained ML model, scaler, and growth model context."""
    if 'time_idx' not in df_predict.columns:
        logger.error("Prediction DataFrame must contain 'time_idx'."); return None
    if ml_model is None: logger.error("ML model not provided."); return None
    if scaler is None: logger.error("Scaler not provided."); return None

    X_pred_time_idx: np.ndarray = df_predict["time_idx"].values.reshape(-1, 1)
    logger.info("Generating ML features for forecasting...")

    context_for_growth_features = X_train_context if X_train_context is not None else X_pred_time_idx
    if X_train_context is None and fitted_growth_models:
        logger.warning("`X_train_context` is None, but growth models provided. Using prediction time indices for growth feature context.")

    try:
        ml_features_df: pd.DataFrame = create_ml_features(
            X_pred=X_pred_time_idx, growth_models_dict=fitted_growth_models, X_fit=context_for_growth_features
        )
    except Exception as e: logger.error(f"Error during ML feature creation: {e}", exc_info=True); return None

    if ml_features_df.empty: logger.error("Feature creation returned empty DataFrame."); return None

    try:
        ml_features_scaled: np.ndarray = scaler.transform(ml_features_df)
    except Exception as e:
        logger.error(f"Error scaling features: {e}. Columns: {ml_features_df.columns.tolist()}", exc_info=True); return None

    logger.info(f"Generating predictions with {type(ml_model).__name__}...")
    try:
        predictions: np.ndarray = ml_model.predict(ml_features_scaled)
    except Exception as e: logger.error(f"Error during model prediction: {e}", exc_info=True); return None

    forecast_output_df = df_predict.copy()
    forecast_output_df["predicted_cumulative"] = predictions
    return forecast_output_df


# --- Main Pipeline Orchestration ---

def main_pipeline(mode: str = "train_and_analyze", data_file: str = "cleaned_enrollments.csv") -> None:
    """Main entry point for training & analysis, or prediction."""
    global fitted_models_global, X_global # Referenced for saving/loading context

    if mode == "train_and_analyze":
        run_training_and_analysis(file_path_str=data_file)

        logger.info("Post-training: Saving key artifacts for standalone prediction...")
        if best_model_name_global and best_model_name_global in fitted_models_global:
            _save_artifact(fitted_models_global[best_model_name_global], f"best_individual_growth_{best_model_name_global}.pkl")

        if fitted_models_global: _save_artifact(fitted_models_global, "all_fitted_growth_models.pkl")
        if X_global is not None: _save_artifact(X_global, "X_train_context_for_growth_models.pkl")
        logger.info("Reminder: `build_ensemble_models` should save 'feature_scaler.pkl' and ML models (e.g., 'rf_model.pkl') for predict mode.")

    elif mode == "predict":
        logger.info("--- Starting Prediction Pipeline ---")
        data_path = PROCESSED_DATA_DIR / data_file
        df_to_predict = load_data(data_path)

        if df_to_predict is None or df_to_predict.empty or 'time_idx' not in df_to_predict.columns:
            logger.error(f"Invalid data for prediction from {data_path}. Must contain 'time_idx'. Exiting."); return

        rf_model, gb_model, loaded_scaler = load_predict_models_and_scaler()
        if loaded_scaler is None: logger.error("Scaler is critical and missing. Prediction aborted."); return

        ml_model_to_use = rf_model if rf_model else gb_model
        if not ml_model_to_use: logger.error("No suitable ML model (RF or GB) loaded. Prediction aborted."); return

        logger.info(f"Using {type(ml_model_to_use).__name__} for prediction.")

        # Load growth models and training X context for feature engineering
        predict_fitted_growth_models = _load_artifact("all_fitted_growth_models.pkl") or {}
        if not predict_fitted_growth_models and fitted_models_global: # Fallback to global
            logger.info("Using growth models from current session's global state (training likely run previously).")
            predict_fitted_growth_models = fitted_models_global

        predict_X_train_context = _load_artifact("X_train_context_for_growth_models.pkl")
        if predict_X_train_context is None and X_global is not None: # Fallback to global
            logger.info("Using X_train_context from current session's global state.")
            predict_X_train_context = X_global

        if not predict_fitted_growth_models: logger.warning("No growth models loaded or found in globals; features from them will be absent.")
        if predict_X_train_context is None and predict_fitted_growth_models : logger.warning("Growth models loaded, but X_train_context is missing; feature generation context might be suboptimal.")

        forecast_df = generate_forecasts(
            df_to_predict, ml_model_to_use, loaded_scaler,
            predict_fitted_growth_models, predict_X_train_context
        )

        if forecast_df is not None:
            base_name = data_file.split('.')[0] if '.' in data_file else data_file
            output_filename = f"forecast_results_{base_name}.csv"
            _save_artifact(forecast_df, output_filename, directory=PROCESSED_DATA_DIR) # Save forecasts
        else:
            logger.error("Forecast generation failed.")
        logger.info("--- Prediction Pipeline Complete ---")
    else:
        logger.error(f"Invalid mode: {mode}. Choose 'train_and_analyze' or 'predict'.")


if __name__ == "__main__":
    # Ensure necessary directories exist
    for_dir in [PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
        for_dir.mkdir(parents=True, exist_ok=True)

    # --- Run Training and Analysis ---
    logger.info("--- Main Execution: Starting Training and Analysis Phase ---")
    # Assumes "cleaned_enrollments.csv" exists or can be generated by `load_data` if it handles that.
    # If `load_data` only loads, the file must be present.
    main_pipeline(mode="train_and_analyze", data_file="cleaned_enrollments.csv")
    logger.info("--- Main Execution: Training and Analysis Phase Complete ---")

    # --- Example: Standalone Prediction ---
    dummy_predict_file_name = "dummy_future_time_data.csv"
    dummy_predict_path = PROCESSED_DATA_DIR / dummy_predict_file_name

    if not dummy_predict_path.exists():
        logger.info(f"Creating dummy data file for prediction example: {dummy_predict_path}")
        start_idx = 0
        # Try to get context from X_global (if training ran in same session) or saved file
        current_X_context = X_global if X_global is not None else _load_artifact("X_train_context_for_growth_models.pkl")
        if current_X_context is not None and isinstance(current_X_context, np.ndarray) and current_X_context.size > 0:
            start_idx = current_X_context[-1] + 1
            logger.info(f"Determined start_idx {start_idx} for dummy data based on training context.")
        else:
            logger.warning("Could not determine training context end for dummy data. Starting time_idx from 0.")

        future_time_indices = np.arange(start_idx, start_idx + 52) # 1 year of weekly data
        dummy_df_predict = pd.DataFrame({'time_idx': future_time_indices})
        dummy_df_predict.to_csv(dummy_predict_path, index=False)
        logger.info(f"Dummy prediction data with 'time_idx' saved to {dummy_predict_path}")

    logger.info(f"\n--- Main Execution: Starting Prediction Phase (using {dummy_predict_file_name}) ---")
    main_pipeline(mode="predict", data_file=dummy_predict_file_name)
    logger.info("--- Main Execution: Prediction Phase Complete ---")
