import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import os
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .utils import plot_manager
from ..utils import generate_bootstrap_predictions
from hiv_enugu.config import FIGURES_DIR, PROCESSED_DATA_DIR

# Set styling for plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")


@plot_manager
def visualize_individual_models(df, y, fitted_models, cv_splits, **kwargs):
    """Visualize and compare individual model performance"""
    fig = plt.figure(figsize=(16, 10))
    train_idx, test_idx = cv_splits[-1]

    plt.scatter(
        df["date"].iloc[train_idx], y[train_idx], color="blue", alpha=0.6, label="Training Data"
    )
    plt.scatter(
        df["date"].iloc[test_idx], y[test_idx], color="red", alpha=0.6, label="Testing Data"
    )

    plt.viridis()

    for i, (name, model) in enumerate(fitted_models.items()):
        y_pred = model["function"](df["time_idx"], *model["parameters"])
        plt.plot(df["date"], y_pred, linewidth=2, label=f"{name} Model")

    plt.title("HIV Growth Models Comparison - Enugu State, Nigeria (2007-2023)", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Number of HIV Patients", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


@plot_manager
def visualize_cumulative_growth(df, y, fitted_models, **kwargs):
    """Visualize and compare cumulative growth models"""
    fig = plt.figure(figsize=(16, 10))

    plt.scatter(df["date"], y, color="blue", alpha=0.6, label="Actual Cumulative Cases")

    plt.viridis()

    for i, (name, model) in enumerate(fitted_models.items()):
        y_pred = model["function"](df["time_idx"], *model["parameters"])
        plt.plot(df["date"], y_pred, linewidth=2, label=f"{name} Model")

    plt.title(
        "Cumulative HIV Growth Models Comparison - Enugu State, Nigeria (2007-2023)", fontsize=16
    )
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Number of HIV Patients", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


@plot_manager
def visualize_ensemble_comparison(
    df, X, y, cv_splits, fitted_models, ensemble_models, best_model_name, **kwargs
):
    """Visualize and compare ensemble models performance"""
    fig = plt.figure(figsize=(16, 10))

    train_idx, test_idx = cv_splits[-1]

    plt.scatter(
        df["date"].iloc[train_idx], y[train_idx], color="blue", alpha=0.6, label="Training Data"
    )
    plt.scatter(
        df["date"].iloc[test_idx], y[test_idx], color="red", alpha=0.6, label="Testing Data"
    )

    plt.plasma()

    best_model = fitted_models[best_model_name]
    y_pred_best = best_model["function"](df["time_idx"], *best_model["parameters"])
    plt.plot(
        df["date"], y_pred_best, linewidth=2, label=f"Best Individual Model: {best_model_name}"
    )

    for i, (name, model) in enumerate(ensemble_models.items()):
        y_pred = model["predict"](X)
        plt.plot(df["date"], y_pred, linewidth=2, label=f"{name} Ensemble")

    plt.title(
        "HIV Growth Ensemble Models Comparison - Enugu State, Nigeria (2007-2023)", fontsize=16
    )
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Number of HIV Patients", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


@plot_manager
def visualize_metrics_comparison(model_metrics, ensemble_metrics, **kwargs):
    """Visualize performance metrics comparison between models"""
    all_metrics = {**model_metrics, **ensemble_metrics}

    metrics_data = {"Model": [], "Type": [], "RMSE": [], "R²": [], "MAE": []}

    for model_name, metrics in all_metrics.items():
        metrics_data["Model"].append(model_name)
        metrics_data["Type"].append("Ensemble" if model_name in ensemble_metrics else "Individual")
        metrics_data["RMSE"].append(metrics.get("RMSE", np.nan))
        metrics_data["R²"].append(metrics.get("R2", np.nan))
        metrics_data["MAE"].append(metrics.get("MAE", np.nan))

    metrics_df = pd.DataFrame(metrics_data)

    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    sns.barplot(x="Model", y="RMSE", hue="Type", data=metrics_df, ax=ax1)
    ax1.set_title("Root Mean Squared Error (RMSE) Comparison", fontsize=14)
    ax1.set_xlabel("")
    ax1.set_ylabel("RMSE (lower is better)", fontsize=12)
    ax1.tick_params(axis="x", rotation=45)

    ax2 = fig.add_subplot(gs[0, 1])
    sns.barplot(x="Model", y="R²", hue="Type", data=metrics_df, ax=ax2)
    ax2.set_title("R-squared (R²) Comparison", fontsize=14)
    ax2.set_xlabel("")
    ax2.set_ylabel("R² (higher is better)", fontsize=12)
    ax2.tick_params(axis="x", rotation=45)

    ax3 = fig.add_subplot(gs[1, 0])
    sns.barplot(x="Model", y="MAE", hue="Type", data=metrics_df, ax=ax3)
    ax3.set_title("Mean Absolute Error (MAE) Comparison", fontsize=14)
    ax3.set_xlabel("")
    ax3.set_ylabel("MAE (lower is better)", fontsize=12)
    ax3.tick_params(axis="x", rotation=45)

    ax4 = fig.add_subplot(gs[1, 1])
    metrics_df["RMSE_norm"] = (metrics_df["RMSE"] - metrics_df["RMSE"].min()) / (
        metrics_df["RMSE"].max() - metrics_df["RMSE"].min() + 1e-10
    )
    metrics_df["R²_norm"] = (metrics_df["R²"] - metrics_df["R²"].min()) / (
        metrics_df["R²"].max() - metrics_df["R²"].min() + 1e-10
    )
    metrics_df["MAE_norm"] = (metrics_df["MAE"] - metrics_df["MAE"].min()) / (
        metrics_df["MAE"].max() - metrics_df["MAE"].min() + 1e-10
    )
    metrics_df["Overall_Score"] = (
        1 - metrics_df["RMSE_norm"] + metrics_df["R²_norm"] + (1 - metrics_df["MAE_norm"])
    ) / 3
    metrics_df = metrics_df.sort_values("Overall_Score", ascending=False)
    sns.barplot(x="Model", y="Overall_Score", hue="Type", data=metrics_df, ax=ax4, dodge=False)
    ax4.set_title("Overall Model Performance Score", fontsize=14)
    ax4.set_xlabel("")
    ax4.set_ylabel("Overall Score (higher is better)", fontsize=12)
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


@plot_manager
def plot_best_model_vs_ensemble(
    df, y, X, best_model, best_ensemble_model, best_model_name, ensemble_name, **kwargs
):
    """Plots the best individual model against the best ensemble model."""
    historical_best_model = best_model["function"](df["time_idx"], *best_model["parameters"])
    historical_ensemble = best_ensemble_model["predict"](X)

    fig = plt.figure(figsize=(16, 8))

    # Plot actual data and predictions
    plt.plot(df["date"], y, color="blue", linewidth=2, label="Actual Cumulative HIV Cases")
    plt.plot(
        df["date"],
        historical_best_model,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Best Individual Model ({best_model_name})",
    )
    plt.plot(
        df["date"],
        historical_ensemble,
        color="purple",
        linestyle="--",
        linewidth=2,
        label=f"Best Ensemble Model ({ensemble_name})",
    )

    # Calculate metrics
    individual_rmse = np.sqrt(mean_squared_error(y, historical_best_model))
    ensemble_rmse = np.sqrt(mean_squared_error(y, historical_ensemble))
    individual_mae = mean_absolute_error(y, historical_best_model)
    ensemble_mae = mean_absolute_error(y, historical_ensemble)
    individual_r2 = r2_score(y, historical_best_model)
    ensemble_r2 = r2_score(y, historical_ensemble)

    # Add metrics annotation
    metrics_text = (
        f"Individual Model Metrics:\n"
        f"RMSE: {individual_rmse:.2f}\n"
        f"MAE: {individual_mae:.2f}\n"
        f"R²: {individual_r2:.3f}\n\n"
        f"Ensemble Model Metrics:\n"
        f"RMSE: {ensemble_rmse:.2f}\n"
        f"MAE: {ensemble_mae:.2f}\n"
        f"R²: {ensemble_r2:.3f}"
    )
    plt.text(
        0.02,
        0.98,
        metrics_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.title("Validation: Actual vs Predicted HIV Cases (2007-2023)", fontsize=16, pad=20)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Number of HIV Patients", fontsize=14)
    plt.legend(loc="center right", bbox_to_anchor=(1.15, 0.5))
    plt.grid(True, alpha=0.3)

    # Improve x-axis formatting
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig


@plot_manager
def plot_forecast(df, y, fitted_models, best_model, best_ensemble_model, ensemble_name, **kwargs):
    """Plots the forecast for future HIV cases."""
    future_dates = pd.to_datetime(pd.date_range(start="2024-01-01", end="2028-12-31", freq="A"))
    max_time_idx = df["time_idx"].max()
    future_time_idx = np.arange(max_time_idx + 1, max_time_idx + 1 + len(future_dates))

    _, lower_bound, upper_bound = generate_bootstrap_predictions(
        best_model["function"], future_time_idx, best_model["parameters"], n_samples=1000
    )

    best_model_future = best_model["function"](future_time_idx, *best_model["parameters"])

    future_features = np.column_stack(
        [
            model["function"](future_time_idx, *model["parameters"])
            for model in fitted_models.values()
        ]
    )

    if ensemble_name in ["Random Forest", "Gradient Boosting"]:
        # Normalize time index for future predictions
        X_max = df["time_idx"].max()
        X_min = df["time_idx"].min()
        future_time_idx_norm = (future_time_idx - X_min) / (X_max - X_min + 1e-9)

        # Create ML features for future predictions
        future_ml_features = np.column_stack(
            [
                future_features,
                future_time_idx_norm,
                np.sin(2 * np.pi * future_time_idx_norm),
                np.cos(2 * np.pi * future_time_idx_norm),
            ]
        )

        scaled_features = best_ensemble_model["scaler"].transform(future_ml_features)
        best_ensemble_future = best_ensemble_model["model"].predict(scaled_features)
    else:
        best_ensemble_future = best_ensemble_model["predict"](future_features)

    future_df = pd.DataFrame(
        {
            "date": future_dates,
            "time_idx": future_time_idx,
            "best_model_forecast": best_model_future,
            "best_ensemble_forecast": best_ensemble_future,
            "lower_ci": lower_bound,
            "upper_ci": upper_bound,
        }
    )

    fig = plt.figure(figsize=(16, 8))
    plt.plot(df["date"], y, color="blue", linewidth=2, label="Historical Data")
    plt.plot(
        future_df["date"],
        future_df["best_model_forecast"],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Forecast (Best Individual Model)",
    )
    plt.plot(
        future_df["date"],
        future_df["best_ensemble_forecast"],
        color="green",
        linestyle="--",
        linewidth=2,
        label="Forecast (Best Ensemble Model)",
    )
    plt.fill_between(
        future_df["date"],
        future_df["lower_ci"],
        future_df["upper_ci"],
        color="red",
        alpha=0.2,
        label="95% Confidence Interval",
    )

    plt.title("Cumulative HIV Trend Forecast (2024-2028) - Enugu State, Nigeria", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Number of HIV Patients", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Ensure the directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    future_df.to_csv(PROCESSED_DATA_DIR / "forecast_results_2024_2028.csv", index=False)
    return fig
