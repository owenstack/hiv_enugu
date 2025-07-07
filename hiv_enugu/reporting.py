from typing import Any, Dict

import numpy as np
import pandas as pd


def generate_coefficient_table(
    fitted_models: Dict[str, Any], model_metrics: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Generates a table of estimated parameters for each standalone growth model in a long format.
    """
    data = []
    param_names_map = {
        "Exponential": ["a", "b", "c"],
        "Logistic": ["L", "k", "t0", "c_offset"],
        "Richards": ["L", "k", "t0", "nu", "c_offset"],
        # Note: Gompertz in fit.py is (L, beta, k, c)
        "Gompertz": ["L", "beta", "k", "c_offset"],
    }

    for name, metrics in model_metrics.items():
        params = metrics.get("parameters")
        if params is not None:
            param_names = param_names_map.get(name, [f"param_{i + 1}" for i in range(len(params))])
            if len(param_names) != len(params):  # Fallback for safety
                param_names = [f"param_{i + 1}" for i in range(len(params))]

            for param_name, param_val in zip(param_names, params):
                data.append({"Model": name, "Parameter": param_name, "Value": param_val})
        else:
            data.append({"Model": name, "Parameter": "Fit Failed", "Value": None})

    df = pd.DataFrame(data)
    return df


def generate_standalone_metrics_table(model_metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Generates a table summarizing performance metrics (RMSE, MAE, R²)
    for each standalone model.

    Args:
        model_metrics: Dictionary of model metrics from fit_growth_models.
                       Example: {'Logistic': {'test_rmse': ..., 'test_r2': ..., 'test_mae': ...}}

    Returns:
        A pandas DataFrame with model names and their performance metrics.
    """
    data = []
    for name, metrics in model_metrics.items():
        # Prioritize test metrics, fallback to train or overall if test not available (e.g. no CV)
        rmse = metrics.get("test_rmse")
        if pd.isna(rmse) and "train_rmse" in metrics:  # Check if test_rmse was NaN (no CV case)
            rmse = metrics.get("train_rmse")

        r2 = metrics.get("test_r2")
        if pd.isna(r2) and "train_r2" in metrics:
            r2 = metrics.get("train_r2")

        mae = metrics.get("test_mae")
        # MAE might not have a 'train_mae' in the current structure, so only test_mae
        # If test_mae is NaN, it will remain NaN.

        data.append(
            {
                "Model": name,
                "RMSE": rmse,
                "R²": r2,
                "MAE": mae,
            }
        )
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    # Example Usage (for testing this module independently)
    dummy_fitted_models = {
        "Logistic": {"function": None, "parameters": [1000, 0.1, 50, 0]},
        "Exponential": {"function": None, "parameters": [100, 0.05, 10]},
    }
    dummy_model_metrics = {
        "Logistic": {
            "parameters": [1000, 0.1, 50, 0],
            "test_rmse": 10.5,
            "test_r2": 0.95,
            "test_mae": 8.2,
            "train_rmse": 9.0,
            "train_r2": 0.97,
        },
        "Exponential": {
            "parameters": [100, 0.05, 10],
            "test_rmse": 15.2,
            "test_r2": 0.90,
            "test_mae": 12.1,
            "train_rmse": 14.0,
            "train_r2": 0.92,
        },
        "Gompertz": {  # Example for a model that might have failed CV
            "parameters": [100, 0.05, 0.2, 10],
            "train_rmse": 12.0,
            "train_r2": 0.93,  # Only train metrics
            "test_rmse": float("nan"),
            "test_r2": float("nan"),
            "test_mae": float("nan"),
        },
    }

    coeff_table = generate_coefficient_table(dummy_fitted_models, dummy_model_metrics)
    print("--- Coefficient Table ---")
    print(coeff_table.to_string())

    metrics_table = generate_standalone_metrics_table(dummy_model_metrics)
    print("\n--- Standalone Metrics Table ---")
    print(metrics_table.to_string())

    # Expected Output for Coefficient Table:
    # --- Coefficient Table ---
    #        Model       L     k    x0    c
    # 0   Logistic  1000.0  0.10  50.0  0.0
    # 1 Exponential   100.0  0.05  10.0  NaN  <- This was for L0, k, c - will adjust param names

    # Expected Output for Metrics Table:
    # --- Standalone Metrics Table ---
    #        Model  RMSE    R²   MAE
    # 0   Logistic  10.5  0.95   8.2
    # 1 Exponential  15.2  0.90  12.1
    # 2    Gompertz  12.0  0.93   NaN

    # Need to adjust param_names_map in generate_coefficient_table based on actual model definitions.
    # Exponential: (L0, k, c) -> a, b, c in `exponential_model(t, a, b, c): return a * np.exp(b * t) + c`
    # Logistic: (L, k, x0, c) -> L, k, t0, c in `logistic_model(t, L, k, t0, c): return c + (L - c) / (1 + np.exp(-k * (t - t0)))`
    # Richards: (L, k, x0, nu, c) -> L, k, t0, nu, c in `richards_model(t, L, k, t0, nu, c): return c + (L - c) / ((1 + nu * np.exp(-k * (t - t0)))**(1/nu))`
    # Gompertz: (L, beta, k, c) -> L, beta, k, c in `gompertz_model(t, L, beta, k, c): return c + (L - c) * np.exp(-beta * np.exp(-k * t))`

    # Corrected param_names_map based on growth_models.py signatures:
    # Exponential: a, b, c
    # Logistic: L, k, t0, c
    # Richards: L, k, t0, nu, c
    # Gompertz: L, beta, k, c
    # The existing map is:
    # Exponential: ["L0", "k", "c"] -> Should be ["a", "b", "c"]
    # Logistic: ["L", "k", "x0", "c"] -> Should be ["L", "k", "t0", "c"] (x0 is often used for t0)
    # Richards: ["L", "k", "x0", "nu", "c"] -> Should be ["L", "k", "t0", "nu", "c"] (x0 for t0)
    # Gompertz: ["L", "beta", "k", "c"] -> Correct
    # I will update this in the actual implementation.
    # pass # Remove pass if it's the last line and functions are added below


def generate_ensemble_input_tables(
    X_time_idx: pd.Series,  # Assuming X_time_idx is a Series/array of the time indices/days
    y_actual: pd.Series,
    fitted_growth_models: Dict[str, Any],  # Contains {'function': callable, 'parameters': list}
    ensemble_models_dict: Dict[str, Any],  # Contains {'predict': callable}
    growth_model_names: list,  # e.g. ['Logistic', 'Richards', ...]
    file_prefix: str = "ensemble_input",
) -> Dict[str, pd.DataFrame]:
    """
    Generates input/prediction tables for each ensemble technique.
    Table format: Day | GrowthModel1_Pred | ... | EnsembleTechnique_Pred

    Args:
        X_time_idx: Time indices or day numbers.
        y_actual: Actual observed values.
        fitted_growth_models: Dictionary of fitted standalone growth models.
        ensemble_models_dict: Dictionary of fitted ensemble models.
        growth_model_names: List of names of the growth models used as input.
        file_prefix: Prefix for the saved CSV files.

    Returns:
        A dictionary where keys are ensemble model names and values are their input DataFrames.
    """
    if X_time_idx.empty or y_actual.empty:
        print("Warning: X_time_idx or y_actual is empty. Cannot generate ensemble input tables.")
        return {}

    # Generate predictions for all growth models once
    growth_model_predictions = {}
    for model_name in growth_model_names:
        if model_name in fitted_growth_models:
            model_info = fitted_growth_models[model_name]
            try:
                growth_model_predictions[model_name] = model_info["function"](
                    X_time_idx.values.ravel(), *model_info["parameters"]
                )
            except Exception as e:
                print(f"Error predicting with {model_name} for ensemble input table: {e}")
                growth_model_predictions[model_name] = np.full(len(X_time_idx), np.nan)
        else:
            growth_model_predictions[model_name] = np.full(len(X_time_idx), np.nan)

    output_tables = {}
    for ensemble_name, ensemble_info in ensemble_models_dict.items():
        df_data = {
            "Day": X_time_idx.values.ravel()
        }  # Use .values to ensure numpy array for older pandas

        # Add growth model predictions
        for gm_name in growth_model_names:
            df_data[f"{gm_name}_Pred"] = growth_model_predictions[gm_name]

        # Add ensemble model's own prediction
        try:
            # The ensemble model's predict function might expect X_time_idx directly
            # or features derived from it.
            # For Simple Average and Weighted Average, they might recalculate based on growth model funcs.
            # For ML models, they take X_time_idx (or features from it).
            # Assuming ensemble_info['predict'] takes X_time_idx (time indices).
            df_data[f"{ensemble_name}_Pred"] = ensemble_info["predict"](X_time_idx.values.ravel())
        except Exception as e:
            print(f"Error predicting with {ensemble_name} for its input table: {e}")
            df_data[f"{ensemble_name}_Pred"] = np.full(len(X_time_idx), np.nan)

        df = pd.DataFrame(df_data)
        output_tables[ensemble_name] = df
        # Optionally save each table here
        # df.to_csv(f"{REPORTS_DIR}/{file_prefix}_{ensemble_name.replace(' ', '_')}.csv", index=False)
        # print(f"Ensemble input table for {ensemble_name} saved.")
    return output_tables


def generate_full_predictions_table(
    X_time_idx: pd.Series,  # Time indices or day numbers
    y_actual: pd.Series,
    fitted_growth_models: Dict[str, Any],
    ensemble_models_dict: Dict[str, Any],
    growth_model_names: list,
) -> pd.DataFrame:
    """
    Generates a table with actual values and predictions from all models.
    Table: Day | Actual | GrowthModel1_Pred | ... | EnsembleModel1_Pred | ...
    """
    if X_time_idx.empty or y_actual.empty:
        print("Warning: X_time_idx or y_actual is empty. Cannot generate full predictions table.")
        return pd.DataFrame()

    df_data = {"Day": X_time_idx.values.ravel(), "Actual": y_actual.values.ravel()}

    # Growth model predictions
    for model_name in growth_model_names:
        if model_name in fitted_growth_models:
            model_info = fitted_growth_models[model_name]
            try:
                df_data[f"{model_name}_Pred"] = model_info["function"](
                    X_time_idx.values.ravel(), *model_info["parameters"]
                )
            except Exception as e:
                print(f"Error predicting with growth model {model_name} for full table: {e}")
                df_data[f"{model_name}_Pred"] = np.full(len(X_time_idx), np.nan)
        else:
            df_data[f"{model_name}_Pred"] = np.full(len(X_time_idx), np.nan)

    # Ensemble model predictions
    for ensemble_name, ensemble_info in ensemble_models_dict.items():
        try:
            df_data[f"{ensemble_name}_Pred"] = ensemble_info["predict"](X_time_idx.values.ravel())
        except Exception as e:
            print(f"Error predicting with ensemble model {ensemble_name} for full table: {e}")
            df_data[f"{ensemble_name}_Pred"] = np.full(len(X_time_idx), np.nan)

    df = pd.DataFrame(df_data)
    return df


def generate_predictions_summary_table(full_predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a summary table (Mean, Std Dev, Min, Max) from the full predictions table.
    Excludes 'Day' and 'Actual' columns from summary statistics.
    """
    if full_predictions_df.empty:
        print("Warning: Full predictions DataFrame is empty. Cannot generate summary.")
        return pd.DataFrame()

    summary_data = []
    # Columns to summarize are all except 'Day' and 'Actual'
    prediction_columns = [
        col for col in full_predictions_df.columns if col not in ["Day", "Actual"]
    ]

    for col in prediction_columns:
        predictions = full_predictions_df[col].dropna()  # Drop NaNs for robust stats
        if not predictions.empty:
            summary_data.append(
                {
                    "Model": col,  # Model name is taken from column header (e.g., "Logistic_Pred")
                    "Mean_Predicted": predictions.mean(),
                    "Std_Dev_Predicted": predictions.std(),
                    "Min_Predicted": predictions.min(),
                    "Max_Predicted": predictions.max(),
                }
            )
        else:
            summary_data.append(
                {
                    "Model": col,
                    "Mean_Predicted": float("nan"),
                    "Std_Dev_Predicted": float("nan"),
                    "Min_Predicted": float("nan"),
                    "Max_Predicted": float("nan"),
                }
            )

    df = pd.DataFrame(summary_data)
    return df


def generate_weighted_average_comparison_table(
    ensemble_metrics: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Generates a table comparing performance of R2 vs InvMSE weighted averages.
    """
    data = []
    # **FIX**: Use the correct metric keys ('test_rmse', 'test_r2', 'test_mae')
    wa_r2_metrics = ensemble_metrics.get("Weighted Average (R2)")
    wa_invmse_metrics = ensemble_metrics.get("Weighted Average (InvMSE)")

    if wa_r2_metrics:
        data.append(
            {
                "Weighting_Method": "R²",
                "RMSE": wa_r2_metrics.get("test_rmse"),
                "R²": wa_r2_metrics.get("test_r2"),
                "MAE": wa_r2_metrics.get("test_mae"),
            }
        )
    if wa_invmse_metrics:
        data.append(
            {
                "Weighting_Method": "Inverse MSE",
                "RMSE": wa_invmse_metrics.get("test_rmse"),
                "R²": wa_invmse_metrics.get("test_r2"),
                "MAE": wa_invmse_metrics.get("test_mae"),
            }
        )

    if not data:
        return pd.DataFrame(columns=["Weighting_Method", "RMSE", "R²", "MAE"])

    return pd.DataFrame(data)


def generate_feature_importance_table(ensemble_models_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Generates a table of feature importances for Random Forest and Gradient Boosting models.
    """
    importance_data = []

    for model_name_key in ["Random Forest", "Gradient Boosting"]:
        if model_name_key in ensemble_models_dict:
            model_info = ensemble_models_dict[model_name_key]
            model_obj = model_info.get("model")
            feature_names = model_info.get("feature_names")

            if hasattr(model_obj, "feature_importances_") and feature_names:
                importances = model_obj.feature_importances_
                for feature, importance in zip(feature_names, importances):
                    importance_data.append(
                        {
                            "Ensemble_Model": model_name_key,
                            "Feature": feature,
                            "Importance": importance,
                        }
                    )
            else:
                print(f"Warning: Feature importances or names not available for {model_name_key}.")

    if not importance_data:
        return pd.DataFrame(columns=["Ensemble_Model", "Feature", "Importance"])

    df = pd.DataFrame(importance_data)
    # Pivot table for better readability: Features as rows, Models as columns for scores
    try:
        pivot_df = df.pivot_table(
            index="Feature", columns="Ensemble_Model", values="Importance"
        ).reset_index()
        pivot_df.fillna(
            0, inplace=True
        )  # Fill NaNs if a feature is not in a model (should not happen with this structure)
        return pivot_df
    except Exception as e:
        print(f"Could not pivot feature importance table: {e}. Returning flat table.")
        return df


def generate_weighted_parameters_table(
    model_metrics: Dict[str, Dict[str, Any]],  # For growth rates
    ensemble_models_dict: Dict[str, Any],  # For weights stored in component_weights
) -> pd.DataFrame:
    """
    Generates a table: Model, Weight (R²), Weight (1/MSE), Growth Rate (k/b).
    """
    data = []
    # Define which parameter index is the growth rate 'k' or 'b' for each model
    # This needs to match the order in model_metrics[model_name]['parameters']
    # From growth_models.py:
    # exponential_model(t, a, b, c) -> b is k (index 1)
    # logistic_model(t, L, k, t0, c) -> k is k (index 1)
    # richards_model(t, L, k, t0, nu, c) -> k is k (index 1)
    # gompertz_model(t, L, beta, k, c) -> k is k (index 2)
    growth_rate_param_indices = {
        "Exponential": 1,  # 'b'
        "Logistic": 1,  # 'k'
        "Richards": 1,  # 'k'
        "Gompertz": 2,  # 'k' (after L and beta)
    }
    # Parameter names for clarity in table (optional)
    growth_rate_param_names = {
        "Exponential": "b (rate)",
        "Logistic": "k (rate)",
        "Richards": "k (rate)",
        "Gompertz": "k (rate)",
    }

    # Get the weights from one of the weighted average models (they should share component_weights)
    # It's stored as: ensemble_models[wa_name]['component_weights'][growth_model_name]['r2_weight' or 'inv_mse_weight']
    component_weights_data = None
    if (
        "Weighted Average (R2)" in ensemble_models_dict
        and "component_weights" in ensemble_models_dict["Weighted Average (R2)"]
    ):
        component_weights_data = ensemble_models_dict["Weighted Average (R2)"]["component_weights"]
    elif (
        "Weighted Average (InvMSE)" in ensemble_models_dict
        and "component_weights" in ensemble_models_dict["Weighted Average (InvMSE)"]
    ):
        # Fallback if R2 version wasn't there for some reason
        component_weights_data = ensemble_models_dict["Weighted Average (InvMSE)"][
            "component_weights"
        ]

    if not component_weights_data:
        print(
            "Warning: Component weights not found in ensemble_models_dict. Cannot generate weighted parameters table."
        )
        return pd.DataFrame(
            columns=[
                "Model",
                "Weight_R2",
                "Weight_InvMSE",
                "Growth_Rate_Param",
                "Growth_Rate_Value",
            ]
        )

    for model_name, metrics_info in model_metrics.items():
        if (
            model_name not in growth_rate_param_indices
        ):  # Skip if not a growth model we have rate info for
            continue

        params = metrics_info.get("parameters")
        growth_rate_val = float("nan")
        if (
            params is not None
            and isinstance(params, (list, np.ndarray))
            and len(params) > growth_rate_param_indices[model_name]
        ):
            growth_rate_val = float(params[growth_rate_param_indices[model_name]])

        param_name_disp = growth_rate_param_names.get(model_name, "N/A")

        # Weights for this specific model_name
        model_specific_weights = component_weights_data.get(model_name, {})
        weight_r2 = model_specific_weights.get("r2_weight", float("nan"))
        weight_inv_mse = model_specific_weights.get("inv_mse_weight", float("nan"))

        data.append(
            {
                "Model": model_name,
                "Weight_R2": weight_r2,
                "Weight_InvMSE": weight_inv_mse,
                "Growth_Rate_Param": param_name_disp,
                "Growth_Rate_Value": growth_rate_val,
            }
        )

    return pd.DataFrame(data)
