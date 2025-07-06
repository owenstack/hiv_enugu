import numpy as np


def create_ml_features(X, fitted_models, X_full):
    """Creates features for machine learning models from individual growth models and time index."""
    time_idx_norm = (X - X_full.min()) / (X_full.max() - X_full.min() + 1e-9)
    base_features = np.column_stack([
            model['function'](X, *model['parameters'])
            for model in fitted_models.values()
        ])
    ml_features = np.column_stack(
        [
            base_features,
            time_idx_norm,
            np.sin(2 * np.pi * time_idx_norm),
            np.cos(2 * np.pi * time_idx_norm),
        ]
    )
    return ml_features
