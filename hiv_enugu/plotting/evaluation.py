import os  # For forecast_future_trends saving CSV

import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from hiv_enugu.config import PROCESSED_DATA_DIR

from .utils import plot_manager

# Set styling for plots (as in user's visualization.py)
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")


@plot_manager
def visualize_individual_models(
    df: pd.DataFrame, X, final_train_idx, final_test_idx, y_train, y_test, fitted_models, **kwargs
):  # Matched signature from user's main() call
    """Visualize individual model fits. df contains original dates for plotting."""
    fig = plt.figure(figsize=(16, 10))

    # Use actual date values from the DataFrame for plotting
    # Ensure df.iloc[X_train_indices] or similar mapping if X_train/X_test are just indices
    # The prompt's main function passes final_train_idx, final_test_idx which are indices for X and y.
    # df['date'] should correspond to the full range of X.

    # Assuming X contains time_idx values that correspond to df rows after processing by load_data
    # We need to map these time_idx values back to dates if df is the original dataframe.
    # However, the main script calls it with df (weekly_df), X (time_idx from weekly_df), y (cumulative from weekly_df)
    # So, df.iloc[final_train_idx]['date'] should work if final_train_idx are positional indices for weekly_df

    plt.scatter(
        df["date"].iloc[final_train_idx], y_train, color="blue", alpha=0.6, label="Training Data"
    )
    plt.scatter(
        df["date"].iloc[final_test_idx], y_test, color="red", alpha=0.6, label="Testing Data"
    )

    colors = ["green", "purple", "orange", "brown", "teal", "magenta"]  # Added more colors
    for i, (name, model) in enumerate(fitted_models.items()):
        y_pred_full_range = model["function"](
            X, *model["parameters"]
        )  # Predict on the full X range
        plt.plot(
            df["date"],
            y_pred_full_range,
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"{name} Model",
        )

    plt.title(
        "Cumulative HIV Growth Models Comparison - Enugu State, Nigeria (2007-2023)", fontsize=16
    )
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Number of HIV Patients", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))  # Show every 2 years
    plt.xticks(rotation=45)
    plt.tight_layout()
    # filename kwarg is handled by plot_manager
    return fig


@plot_manager
def visualize_ensemble_comparison(
    df: pd.DataFrame,
    X,
    y,
    cv_splits,
    fitted_models,
    ensemble_models_dict,
    best_model_name,
    **kwargs,
):  # Renamed ensemble_models to ensemble_models_dict
    """Visualize and compare ensemble models performance."""
    fig = plt.figure(figsize=(16, 10))

    # Use the last CV split for train/test visualization
    train_idx, test_idx = cv_splits[-1]

    plt.scatter(
        df["date"].iloc[train_idx],
        y[train_idx],
        color="blue",
        alpha=0.6,
        label="Training Data (last fold)",
    )
    plt.scatter(
        df["date"].iloc[test_idx],
        y[test_idx],
        color="red",
        alpha=0.6,
        label="Testing Data (last fold)",
    )

    # Plot best individual model
    if best_model_name in fitted_models:
        best_individual_model_details = fitted_models[best_model_name]
        y_pred_best_individual = best_individual_model_details["function"](
            X, *best_individual_model_details["parameters"]
        )
        plt.plot(
            df["date"],
            y_pred_best_individual,
            color="green",
            linewidth=2,
            linestyle="--",
            label=f"Best Individual ({best_model_name})",
        )

    colors = [
        "purple",
        "orange",
        "brown",
        "magenta",
        "cyan",
        "lime",
        "pink",  # Added more colors for potentially more models
        "gray",
        "olive",
        "navy",
    ]  # Cycle through these for ensembles

    # Dynamically plot all models from ensemble_models_dict
    # Sort to maintain a somewhat consistent plotting order if desired, e.g., alphabetically
    # Or define a preferred order if that's important. For now, iterating directly.
    plot_order = [
        "Simple Average",
        "Weighted Average (R2)",
        "Weighted Average (InvMSE)",
        "Random Forest",
        "Gradient Boosting",
    ]

    # Plot models in defined order first, then any others
    models_plotted_count = 0
    for name in plot_order:
        if name in ensemble_models_dict:
            model_info = ensemble_models_dict[name]
            y_pred_ensemble = model_info["predict"](X)
            plt.plot(
                df["date"],
                y_pred_ensemble,
                color=colors[models_plotted_count % len(colors)],
                linewidth=2,
                label=f"{name} Ensemble",
            )
            models_plotted_count += 1

    # Plot any remaining models not in plot_order
    for i, (name, model_info) in enumerate(ensemble_models_dict.items()):
        if name not in plot_order:  # Check if already plotted
            # The 'predict' function in ensemble_models_dict expects X (time_idx)
            y_pred_ensemble = model_info["predict"](X)
            plt.plot(
                df["date"],
                y_pred_ensemble,
                color=colors[i % len(colors)],
                linewidth=2,
                label=f"{name} Ensemble",
            )
        else:
            print(
                f"Warning: Ensemble model '{name}' not found in ensemble_models_dict for plotting."
            )

    plt.title(
        "HIV Growth Ensemble Models Comparison - Enugu State, Nigeria", fontsize=16
    )  # Removed date range, dynamic
    plt.xlabel("Date", fontsize=14)
    plt.ylabel(
        "Cumulative Number of HIV Patients", fontsize=14
    )  # Changed from "Number of HIV Patients"
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    # filename kwarg is handled by plot_manager
    return fig


@plot_manager
def visualize_metrics_comparison(
    model_metrics, ensemble_metrics_dict, **kwargs
):  # Renamed ensemble_metrics
    """Visualize performance metrics comparison between models."""
    all_metrics_combined = {**model_metrics, **ensemble_metrics_dict}  # Combine dicts

    metrics_data = {"Model": [], "Type": [], "RMSE": [], "R²": [], "MAE": []}

    for model_name, metrics_values in all_metrics_combined.items():
        metrics_data["Model"].append(model_name)
        # Determine type based on which original dictionary it came from
        if model_name in model_metrics:
            metrics_data["Type"].append("Individual")
            # Assuming keys like 'test_rmse', 'test_r2', 'test_mae' from fit_growth_models
            metrics_data["RMSE"].append(metrics_values.get("test_rmse", np.nan))
            metrics_data["R²"].append(metrics_values.get("test_r2", np.nan))
            metrics_data["MAE"].append(metrics_values.get("test_mae", np.nan))
        elif model_name in ensemble_metrics_dict:
            metrics_data["Type"].append("Ensemble")
            # Assuming keys like 'test_rmse', 'test_r2', 'test_mae' from build_ensemble_models (CV averages)
            metrics_data["RMSE"].append(
                metrics_values.get("test_rmse", np.nan)
            )  # Or just 'rmse' if that's the key
            metrics_data["R²"].append(metrics_values.get("test_r2", np.nan))  # Or just 'r2'
            metrics_data["MAE"].append(metrics_values.get("test_mae", np.nan))  # Or just 'mae'
        else:  # Should not happen if logic is correct
            metrics_data["Type"].append("Unknown")
            metrics_data["RMSE"].append(np.nan)
            metrics_data["R²"].append(np.nan)
            metrics_data["MAE"].append(np.nan)

    metrics_df = pd.DataFrame(metrics_data)

    fig = plt.figure(figsize=(18, 15))  # Original size
    gs = GridSpec(2, 2, figure=fig)  # Original layout

    ax1 = fig.add_subplot(gs[0, 0])
    sns.barplot(x="Model", y="RMSE", hue="Type", data=metrics_df, ax=ax1, dodge=True)
    ax1.set_title("Test Root Mean Squared Error (RMSE) Comparison", fontsize=14)
    ax1.set_xlabel("")
    ax1.set_ylabel("RMSE (lower is better)", fontsize=12)

    plt.setp(ax1.get_xticklabels(), rotation=45)

    ax2 = fig.add_subplot(gs[0, 1])
    sns.barplot(x="Model", y="R²", hue="Type", data=metrics_df, ax=ax2, dodge=True)
    ax2.set_title("Test R² Score Comparison", fontsize=14)
    ax2.set_xlabel("")
    ax2.set_ylabel("R² (higher is better)", fontsize=12)

    plt.setp(ax2.get_xticklabels(), rotation=45)

    ax3 = fig.add_subplot(gs[1, 0])
    sns.barplot(x="Model", y="MAE", hue="Type", data=metrics_df, ax=ax3, dodge=True)
    ax3.set_title("Test Mean Absolute Error (MAE) Comparison", fontsize=14)
    ax3.set_xlabel("")
    ax3.set_ylabel("MAE (lower is better)", fontsize=12)

    plt.setp(ax3.get_xticklabels(), rotation=45)

    # Overall Score Plot (as in user's code)
    ax4 = fig.add_subplot(gs[1, 1])
    metrics_df_copy = metrics_df.copy()  # Avoid SettingWithCopyWarning
    metrics_df_copy.dropna(
        subset=["RMSE", "R²", "MAE"], inplace=True
    )  # Drop rows where essential metrics are NaN for scoring

    if not metrics_df_copy.empty:
        # Normalize: smaller is better for RMSE, MAE; larger for R2
        rmse_min, rmse_max = metrics_df_copy["RMSE"].min(), metrics_df_copy["RMSE"].max()
        mae_min, mae_max = metrics_df_copy["MAE"].min(), metrics_df_copy["MAE"].max()
        r2_min, r2_max = metrics_df_copy["R²"].min(), metrics_df_copy["R²"].max()

        metrics_df_copy["RMSE_norm"] = (metrics_df_copy["RMSE"] - rmse_min) / (
            rmse_max - rmse_min + 1e-10
        )
        metrics_df_copy["MAE_norm"] = (metrics_df_copy["MAE"] - mae_min) / (
            mae_max - mae_min + 1e-10
        )
        # For R², higher is better, so (R² - min) / (max - min) gives higher scores for better R²
        metrics_df_copy["R²_norm_score"] = (metrics_df_copy["R²"] - r2_min) / (
            r2_max - r2_min + 1e-10
        )

        # Overall score: (1 - RMSE_norm) + R²_norm_score + (1 - MAE_norm)
        # Max score is 3, min is 0.
        metrics_df_copy["Overall_Score"] = (
            (1 - metrics_df_copy["RMSE_norm"])
            + metrics_df_copy["R²_norm_score"]
            + (1 - metrics_df_copy["MAE_norm"])
        )
        metrics_df_copy["Overall_Score"] = (
            metrics_df_copy["Overall_Score"] / 3.0
        )  # Normalize to 0-1 range

        metrics_df_copy = metrics_df_copy.sort_values("Overall_Score", ascending=False)
        sns.barplot(
            x="Model", y="Overall_Score", hue="Type", data=metrics_df_copy, ax=ax4, dodge=True
        )
    else:
        ax4.text(
            0.5,
            0.5,
            "Not enough data for Overall Score",
            horizontalalignment="center",
            verticalalignment="center",
        )

    ax4.set_title("Overall Model Performance Score (Normalized)", fontsize=14)
    ax4.set_xlabel("")
    ax4.set_ylabel("Performance Score (higher is better)", fontsize=12)

    plt.setp(ax4.get_xticklabels(), rotation=45)

    plt.tight_layout(rect=(0, 0, 1, 0.97))  # Adjust rect to prevent suptitle overlap if added
    fig.suptitle("Model Performance Metrics Comparison", fontsize=18, y=0.99)
    # filename kwarg is handled by plot_manager
    return fig, metrics_df  # Return the DataFrame as well


@plot_manager
def create_validation_plot(
    df,
    X,
    y,
    best_individual_model_details,
    best_ensemble_model_info,
    ensemble_name,
    best_model_name,
    **kwargs,
):
    """Create validation plot comparing actual vs predicted values for historical data."""
    # Predictions for best individual model
    historical_best_individual_pred = best_individual_model_details["function"](
        X, *best_individual_model_details["parameters"]
    )

    # Predictions for best ensemble model
    # The predict function in best_ensemble_model_info expects X (time_idx)
    historical_ensemble_pred = best_ensemble_model_info["predict"](X)

    fig = plt.figure(figsize=(16, 8))

    plt.plot(
        df["date"], y, color="blue", linewidth=2.5, alpha=0.8, label="Actual Cumulative HIV Cases"
    )

    plt.plot(
        df["date"],
        historical_best_individual_pred,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Best Individual: {best_model_name}",
    )
    plt.plot(
        df["date"],
        historical_ensemble_pred,
        color="purple",
        linestyle=":",
        linewidth=2,
        label=f"Best Ensemble: {ensemble_name}",
    )

    # Metrics for display
    individual_rmse = np.sqrt(mean_squared_error(y, historical_best_individual_pred))
    ensemble_rmse = np.sqrt(mean_squared_error(y, historical_ensemble_pred))
    individual_mae = mean_absolute_error(y, historical_best_individual_pred)
    ensemble_mae = mean_absolute_error(y, historical_ensemble_pred)
    individual_r2 = r2_score(y, historical_best_individual_pred)
    ensemble_r2 = r2_score(y, historical_ensemble_pred)

    plt.title("Validation: Actual vs. Predicted HIV Cases (Full Historical Data)", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Number of HIV Patients", fontsize=14)
    plt.legend(fontsize=12, loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)

    # Adding text box for metrics, improved placement
    metrics_text = (
        f"--- Best Individual ({best_model_name}) ---\n"
        f"RMSE: {individual_rmse:.2f}, MAE: {individual_mae:.2f}, R²: {individual_r2:.3f}\n"
        f"--- Best Ensemble ({ensemble_name}) ---\n"
        f"RMSE: {ensemble_rmse:.2f}, MAE: {ensemble_mae:.2f}, R²: {ensemble_r2:.3f}"
    )
    plt.figtext(
        0.5,
        0.01,
        metrics_text,
        ha="center",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.7, "pad": 5},
    )

    plt.tight_layout(rect=(0, 0.05, 1, 0.95))  # Adjust rect for figtext and title
    # filename kwarg is handled by plot_manager
    return fig


@plot_manager
def visualize_single_model_fit(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    model_details: dict,
    final_train_idx: np.ndarray,
    final_test_idx: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    **kwargs,
):
    """
    Visualize a single model's fit against observed data (training and testing).

    Args:
        df: DataFrame with 'date' column corresponding to X and y.
        X: Array of time indices for the full dataset.
        y: Array of actual values for the full dataset.
        model_name: Name of the model (e.g., "Logistic").
        model_details: Dictionary containing 'function' and 'parameters' for the model.
        final_train_idx: Indices for training data points in X and y.
        final_test_idx: Indices for testing data points in X and y.
        y_train: Training actual values.
        y_test: Testing actual values.
        **kwargs: Passed to plot_manager, including 'filename'.
    """
    fig = plt.figure(figsize=(12, 7))

    # Plot training and testing data points
    plt.scatter(
        df["date"].iloc[final_train_idx], y_train, color="blue", alpha=0.6, label="Training Data"
    )
    plt.scatter(
        df["date"].iloc[final_test_idx], y_test, color="red", alpha=0.6, label="Testing Data"
    )

    # Predict on the full X range for the specific model
    y_pred_full_range = model_details["function"](X, *model_details["parameters"])
    plt.plot(
        df["date"],
        y_pred_full_range,
        color="green",
        linewidth=2,
        label=f"{model_name} Model Fit",
    )

    # Calculate metrics for this model on the full dataset for display
    rmse_full = np.sqrt(mean_squared_error(y, y_pred_full_range))
    mae_full = mean_absolute_error(y, y_pred_full_range)
    r2_full = r2_score(y, y_pred_full_range)

    metrics_text = f"RMSE: {rmse_full:.2f}\nMAE: {mae_full:.2f}\nR²: {r2_full:.3f}"
    # Position the text box; adjust as needed
    plt.text(
        0.05,
        0.95,
        metrics_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
    )

    plt.title(
        f"Observed vs. Predicted: {model_name} Model\nEnugu State HIV Data (2007-2023)",
        fontsize=16,
    )
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Number of HIV Patients", fontsize=14)
    plt.legend(fontsize=12, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


@plot_manager
def forecast_future_trends(
    df,
    X,
    y,
    best_individual_model_details,
    best_ensemble_model_info,
    ensemble_name,
    fitted_models,
    best_model_name,
    generate_bootstrap_predictions_func,
    **kwargs,
):
    """Forecast future HIV trends with enhanced uncertainty quantification."""
    # Determine future date range and time indices
    # last_date = df["date"].max() # This variable was unused
    # Ensure future_dates are daily for smoother plotting if needed, then resample/select for specific points if required
    # future_dates_daily = pd.date_range( # This variable was unused
    #     start=last_date + pd.Timedelta(days=1), periods=5 * 365, freq="D" # original line used last_date
    # )

    # time_idx for future dates
    # The X passed is the original time_idx for training. Use its min/max for consistency.
    # And its last value to continue the sequence.
    max_train_time_idx = X.max()
    # Future time_idx should be daily if future_dates_daily is used for prediction.
    # If models expect weekly time_idx, then this needs adjustment or models handle it.
    # The original code implies future_time_idx is also weekly-based continuation.
    # Let's assume weekly continuation from max_train_time_idx for model prediction inputs.
    # (5 years * 52 weeks/year approx)
    num_future_weeks = 5 * 52
    future_time_idx_weekly = np.arange(
        max_train_time_idx + 1, max_train_time_idx + 1 + num_future_weeks
    )

    # For plotting, we need dates corresponding to these weekly time indices.
    # Start from the week after the last date in df.
    last_week_date = df["date"].max()
    future_plot_dates_weekly = pd.date_range(
        start=last_week_date + pd.Timedelta(weeks=1), periods=num_future_weeks, freq="W-SUN"
    )  # Or matching df's week frequency

    # Bootstrap predictions for the best individual model
    # generate_bootstrap_predictions_func is passed, should be the actual utility function
    _, lower_bound, upper_bound = generate_bootstrap_predictions_func(
        best_individual_model_details["function"],
        future_time_idx_weekly,  # Use weekly time_idx for prediction
        best_individual_model_details["parameters"],
        n_samples=1000,  # As in original prompt
    )

    best_model_future_pred = best_individual_model_details["function"](
        future_time_idx_weekly, *best_individual_model_details["parameters"]
    )

    # Predictions for best ensemble model
    # The 'predict' function in best_ensemble_model_info expects X (time_idx)
    best_ensemble_future_pred = best_ensemble_model_info["predict"](
        future_time_idx_weekly
    )  # Use weekly time_idx

    # Create DataFrame for forecast results
    future_results_df = pd.DataFrame(
        {
            "date": future_plot_dates_weekly,  # Dates for plotting
            "time_idx": future_time_idx_weekly,  # Corresponding time_idx
            "best_model_forecast": best_model_future_pred,
            "best_ensemble_forecast": best_ensemble_future_pred,
            "lower_bound_ci": lower_bound,  # CI for the best individual model
            "upper_bound_ci": upper_bound,
        }
    )

    fig = plt.figure(figsize=(16, 8))

    # Plot historical data
    plt.plot(
        df["date"], y, color="blue", alpha=0.8, linewidth=2, label="Historical Data"
    )  # Changed from scatter

    # Plot historical fit of the best individual model for context
    historical_fitted_best_individual = best_individual_model_details["function"](
        X, *best_individual_model_details["parameters"]
    )
    plt.plot(
        df["date"],
        historical_fitted_best_individual,
        color="darkgreen",
        linestyle="-",
        linewidth=1.5,
        label=f"Historical Fit ({best_model_name})",
    )

    # Plot forecasts
    plt.plot(
        future_results_df["date"],
        future_results_df["best_model_forecast"],
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Forecast ({best_model_name})",
    )
    plt.plot(
        future_results_df["date"],
        future_results_df["best_ensemble_forecast"],
        color="purple",
        linestyle=":",
        linewidth=2,
        label=f"Forecast ({ensemble_name} Ensemble)",
    )

    # Plot confidence interval for the best individual model's forecast
    plt.fill_between(
        future_results_df["date"],
        future_results_df["lower_bound_ci"],
        future_results_df["upper_bound_ci"],
        color="green",
        alpha=0.2,
        label="95% CI (Indiv. Model)",
    )

    plt.title("Cumulative HIV Trend Forecast (Next 5 Years) - Enugu State, Nigeria", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Number of HIV Patients", fontsize=14)
    plt.legend(fontsize=12, loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Show year
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))  # Tick every year
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save forecast data to CSV (as in user's code)
    # Ensure 'data' directory exists, or save to 'reports/data/' or similar
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    forecast_csv_path = os.path.join(
        PROCESSED_DATA_DIR, "forecast_results_next_5_years.csv"
    )  # Modified filename slightly
    future_results_df.to_csv(forecast_csv_path, index=False)
    print(f"Forecast data saved to {forecast_csv_path}")
    # filename kwarg for plot is handled by plot_manager
    return fig


@plot_manager
def plot_feature_importances(
    ensemble_models_dict: dict,  # Full dictionary of all ensemble models
    model_name_key: str,  # Specific model to plot (e.g., "Random Forest")
    top_n: int = 15,
    **kwargs,  # For plot_manager (filename) and model_display_name
):
    """
    Creates and saves a bar plot of feature importances for a specified ML model.
    Uses the 'feature_importances_' attribute and 'feature_names' from the model dictionary.
    This function is designed to be called once per model (e.g., once for RF, once for GB).
    """
    if model_name_key not in ensemble_models_dict:
        print(
            f"Model '{model_name_key}' not found in ensemble_models_dict. Cannot plot feature importances."
        )
        # To prevent plot_manager from erroring on None, maybe return an empty fig or handle upstream
        # For now, let pipeline handle the check before calling. Or plot_manager should handle None.
        # If we must return a fig: fig = plt.figure(); plt.text(0.5,0.5, "Model not found"); return fig
        return None

    model_info = ensemble_models_dict[model_name_key]
    model_obj = model_info.get("model")
    feature_names = model_info.get("feature_names")

    if not (model_obj and hasattr(model_obj, "feature_importances_") and feature_names):
        print(f"Feature importances or names not available for '{model_name_key}'. Skipping plot.")
        return None

    importances = model_obj.feature_importances_
    if not isinstance(importances, np.ndarray) or not importances.size > 0:
        print(
            f"Importances for {model_name_key} are not a valid array or are empty. Skipping plot."
        )
        return None

    named_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    # Ensure top_n doesn't exceed available features
    actual_top_n = min(top_n, len(named_importances))
    if actual_top_n == 0:
        print(f"No features with importance found for {model_name_key}. Skipping plot.")
        return None

    top_features = [item[0] for item in named_importances[:actual_top_n]]
    top_importances = [item[1] for item in named_importances[:actual_top_n]]

    fig = plt.figure(figsize=(12, max(6, actual_top_n * 0.45)))  # Adjust height
    # Using hue for individual colors per bar, but then hiding legend as it's redundant.
    sns.barplot(
        x=top_importances,
        y=top_features,
        palette="viridis",
        hue=top_features,
        dodge=False,
        legend=False,
    )

    display_name = kwargs.get(
        "model_display_name", model_name_key
    )  # Use passed display name or the key
    plt.title(f"Top {actual_top_n} Feature Importances: {display_name}", fontsize=16)
    plt.xlabel("Importance Score", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.gca().invert_yaxis()  # Display most important at the top
    plt.tight_layout()

    return fig  # plot_manager decorator will save this using 'filename' from kwargs


@plot_manager
def visualize_weighted_average_metrics_comparison(
    weighted_avg_comparison_df: pd.DataFrame, **kwargs
):
    """
    Visualizes the comparison of performance metrics for R² and Inverse MSE weighted averages.

    Args:
        weighted_avg_comparison_df: DataFrame from generate_weighted_average_comparison_table.
                                    Expected columns: "Weighting_Method", "RMSE", "R²", "MAE".
        **kwargs: Passed to plot_manager, including 'filename'.
    """
    if weighted_avg_comparison_df.empty:
        fig = plt.figure()
        plt.text(0.5, 0.5, "No data for weighted average comparison.", ha="center", va="center")
        return fig

    df_melted = weighted_avg_comparison_df.melt(
        id_vars=["Weighting_Method"],
        value_vars=["RMSE", "R²", "MAE"],
        var_name="Metric",
        value_name="Score",
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Weighted Average Ensemble: R² vs. Inverse MSE Weighting Performance", fontsize=16
    )

    metrics_to_plot = ["RMSE", "R²", "MAE"]
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        metric_data = df_melted[df_melted["Metric"] == metric]
        if not metric_data.empty:
            sns.barplot(
                x="Weighting_Method",
                y="Score",
                data=metric_data,
                ax=ax,
                palette="pastel",
                hue="Weighting_Method",
                dodge=False,
                legend=False,
            )
            ax.set_title(f"{metric} Comparison", fontsize=14)
            ax.set_xlabel("Weighting Method", fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.tick_params(axis="x", rotation=15)
            for container in ax.containers:
                ax.bar_label(container, fmt="%.3f")
        else:
            ax.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center")

    plt.tight_layout(rect=(0, 0, 1, 0.96))  # Adjust layout to make space for suptitle
    return fig
