import os
import matplotlib.pyplot as plt
from hiv_enugu.config import FIGURES_DIR


def plot_manager(plot_func):
    """Decorator for plot functions to handle saving and closing plots"""

    def wrapper(*args, **kwargs):
        fig = plot_func(*args, **kwargs)

        # Create plots directory if it doesn't exist
        os.makedirs(FIGURES_DIR, exist_ok=True)

        filename = kwargs.get("filename", f"{plot_func.__name__}.png")
        plt.savefig(os.path.join(FIGURES_DIR, filename))
        plt.close(fig)
        return fig

    return wrapper
