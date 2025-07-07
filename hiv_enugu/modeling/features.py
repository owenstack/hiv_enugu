import numpy as np
import pandas as pd


def create_ml_features(X, fitted_models, X_full):
    """
    Creates features for machine learning models from individual growth models and time index.
    Returns a pandas DataFrame with named features.
    """
    # Ensure X is a numpy array for calculations if it's a pandas Series/DataFrame index
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        X_np = np.asarray(X).flatten()
    else:
        X_np = np.asarray(X).flatten()

    if isinstance(X_full, pd.Series) or isinstance(X_full, pd.DataFrame):
        X_full_np = np.asarray(X_full).flatten()
    else:
        X_full_np = np.asarray(X_full).flatten()

    # Handle empty X or X_full to prevent division by zero or min/max errors
    if len(X_full_np) == 0:
        time_idx_norm = np.zeros_like(X_np, dtype=float)  # Avoid NaN if X_full is empty
    else:
        x_full_min = X_full_np.min()
        x_full_max = X_full_np.max()
        denominator = x_full_max - x_full_min
        if denominator == 0:  # Avoid division by zero if all X_full values are the same
            time_idx_norm = np.zeros_like(X_np, dtype=float)
        else:
            time_idx_norm = (X_np - x_full_min) / (denominator + 1e-9)  # Added 1e-9 from original

    # Create base features from fitted models
    feature_columns = []
    column_names = []

    for model_name, model_details in fitted_models.items():
        predictions = model_details["function"](X_np, *model_details["parameters"])
        feature_columns.append(predictions)
        column_names.append(model_name)

    # Add time-based features
    feature_columns.append(time_idx_norm)
    column_names.append("time_norm")

    feature_columns.append(np.sin(2 * np.pi * time_idx_norm))
    column_names.append("time_norm_sin")

    feature_columns.append(np.cos(2 * np.pi * time_idx_norm))
    column_names.append("time_norm_cos")

    # Stack features horizontally to form the DataFrame data
    # Ensure all columns are 1D arrays of the same length
    # If X_np is scalar, this needs adjustment, but X is typically an array of time points.
    if X_np.ndim == 0:  # If X is scalar
        ml_features_array = np.array(
            [fc if np.isscalar(fc) else fc[0] for fc in feature_columns]
        ).reshape(1, -1)
    else:
        ml_features_array = np.column_stack(feature_columns)

    ml_features_df = pd.DataFrame(ml_features_array, columns=column_names)

    return ml_features_df
