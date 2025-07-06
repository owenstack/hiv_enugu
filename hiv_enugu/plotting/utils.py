import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from hiv_enugu.config import FIGURES_DIR


def plot_manager(plot_func):
    """Decorator for plot functions to handle saving and closing plots"""

    def wrapper(*args, **kwargs):
        fig = plot_func(*args, **kwargs)
        os.makedirs(FIGURES_DIR, exist_ok=True)

        filename = kwargs.get("filename")
        if filename:
            plt.savefig(os.path.join(FIGURES_DIR, filename))
        else:
            # Default filename if not provided, using the function's name
            plt.savefig(os.path.join(FIGURES_DIR, f"{plot_func.__name__}.png"))

        plt.close(fig)
        return fig

    return wrapper
