import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats # Added for plot_qq
from .utils import plot_manager # Assuming utils.py is in the same directory (plotting)

# Global style settings (optional, can be set in main script or per plot)
# plt.style.use('seaborn-v0_8-whitegrid')
# sns.set_palette('Set2')

@plot_manager
def plot_residuals(y_true, y_pred, model_name, filename_suffix="", **kwargs): # Added filename_suffix and **kwargs
    """Plot residuals vs. predicted values."""
    residuals = y_true - y_pred
    fig = plt.figure(figsize=(10, 6)) # Create a new figure for each plot
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel(f'Predicted Values ({model_name})')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot for {model_name}{filename_suffix}') # Include suffix in title
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Filename is passed via kwargs to plot_manager by the calling function (e.g., fit.py)
    return fig

@plot_manager
def plot_qq(residuals, model_name, filename_suffix="", **kwargs): # Added filename_suffix and **kwargs
    """Create a Q-Q plot of residuals."""
    fig = plt.figure(figsize=(8, 8)) # Create a new figure
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of Residuals for {model_name}{filename_suffix}') # Include suffix in title
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Filename is passed via kwargs to plot_manager
    return fig

@plot_manager
def plot_residuals_histogram(residuals, model_name, filename_suffix="", **kwargs): # Changed y_true, y_pred to residuals, added filename_suffix and **kwargs
    """Create a histogram of residuals."""
    # residuals = y_true - y_pred # Residuals are now passed directly
    fig = plt.figure(figsize=(10, 6)) # Create a new figure
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Residuals for {model_name}{filename_suffix}') # Include suffix in title
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Filename is passed via kwargs to plot_manager
    return fig
