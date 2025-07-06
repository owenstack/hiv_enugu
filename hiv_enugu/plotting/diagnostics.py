import matplotlib.pyplot as plt
import seaborn as sns
from .utils import plot_manager


@plot_manager
def plot_residuals(y_true, y_pred, model_name, **kwargs):
    """Plot residuals vs. predicted values."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel(f'Predicted Values ({model_name})')
    ax.set_ylabel('Residuals')
    ax.set_title(f'Residual Plot for {model_name}')
    return fig


@plot_manager
def plot_residuals_histogram(y_true, y_pred, model_name, **kwargs):
    """Plot histogram of residuals."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Histogram of Residuals for {model_name}')
    return fig
