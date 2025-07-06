import matplotlib.pyplot as plt
import seaborn as sns
from .utils import plot_manager


@plot_manager
def plot_residuals(y_true, y_pred, model_name, filename_suffix="", **kwargs):
    """Plot residuals vs. predicted values."""
    residuals = y_true - y_pred
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel(f"Predicted Values ({model_name})")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot for {model_name}")
    plt.grid(True)
    plt.tight_layout()
    if "filename" not in kwargs:
        kwargs["filename"] = f"residuals_plot_{model_name}{filename_suffix}.png"
    return fig


@plot_manager
def plot_residuals_histogram(y_true, y_pred, model_name, filename_suffix="", **kwargs):
    """Plot histogram of residuals."""
    residuals = y_true - y_pred
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Residuals for {model_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if "filename" not in kwargs:
        kwargs["filename"] = f"residuals_histogram_{model_name}{filename_suffix}.png"
    return fig
