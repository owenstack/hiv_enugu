import os

import matplotlib.pyplot as plt

from hiv_enugu.config import FIGURES_DIR


def plot_manager(plot_func):
    """Decorator for plot functions to handle saving and closing plots"""

    def wrapper(*args, **kwargs):
        result = plot_func(*args, **kwargs)
        os.makedirs(FIGURES_DIR, exist_ok=True)

        # Handle both single figure and tuple returns
        if isinstance(result, tuple):
            fig = result[0]
            other_data = result[1:]
        else:
            fig = result
            other_data = ()

        filename = kwargs.get("filename")
        if filename:
            plt.savefig(os.path.join(FIGURES_DIR, filename))
        else:
            # Default filename if not provided, using the function's name
            plt.savefig(os.path.join(FIGURES_DIR, f"{plot_func.__name__}.png"))

        plt.close(fig)

        # Return original result structure
        if other_data:
            return (fig, *other_data)
        else:
            return fig

    return wrapper
