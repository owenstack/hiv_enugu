import matplotlib.pyplot as plt
# import seaborn as sns # Not used in this specific file, but often used with plotting
import numpy as np
import pandas as pd
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Not used here
# import scipy.stats as stats # Not used here
import os
# from utilities import generate_bootstrap_predictions # This seems to be a project specific utility, not for plotting utils

# Set styling for plots - This could be a global setting or applied per plot
# plt.style.use('seaborn-v0_8-whitegrid') # Consider where to best set style
# sns.set_palette('Set2') # Consider where to best set palette

def plot_manager(plot_func):
    """Decorator for plot functions to handle saving and closing plots"""
    def wrapper(*args, **kwargs):
        fig = plot_func(*args, **kwargs)

        # Create plots directory if it doesn't exist
        # Aligning with the user-provided code's structure
        output_dir = 'plots'
        os.makedirs(output_dir, exist_ok=True)

        filename = kwargs.get("filename")
        if filename:
            plt.savefig(os.path.join(output_dir, filename))
        else:
            # Default filename if not provided, using the function's name
            plt.savefig(os.path.join(output_dir, f"{plot_func.__name__}.png"))

        plt.close(fig)
        return fig
    return wrapper
