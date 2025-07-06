import numpy as np
import pandas as pd  # Though not directly used in prepare_data_for_modeling, often useful in context
from sklearn.model_selection import TimeSeriesSplit


def prepare_data_for_modeling(df: pd.DataFrame, n_splits=5):
    """Prepare data for model fitting with full cross-validation"""
    # Extract features
    # Ensure 'time_idx' and 'cumulative' columns exist in the DataFrame df
    if "time_idx" not in df.columns:
        raise ValueError("DataFrame must contain a 'time_idx' column.")
    if "cumulative" not in df.columns:
        raise ValueError("DataFrame must contain a 'cumulative' column.")

    X = np.array(df["time_idx"].values)
    y = np.array(df["cumulative"].values)

    # Validate data
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        # Try to handle internal NaNs by forward fill, if appropriate
        # This assumes that cumulative data should generally persist if a value is missing
        y_series = pd.Series(y)
        y_series = y_series.fillna(method="ffill") # type: ignore
        y_series = y_series.fillna(method="bfill")  # For leading NaNs
        y = y_series.values
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError(
                "Invalid values (NaN or Inf) detected in target variable 'cumulative' even after attempting ffill/bfill."
            )

    if len(y) < 52:  # At least 1 year of weekly data
        print(
            f"Warning: Insufficient data points for modeling (found {len(y)}, expected at least 52). Model performance may be affected."
        )
        # Reducing n_splits if data is very short
        if len(y) < 10 and n_splits > 2:  # Arbitrary small number
            n_splits = 2
        elif (
            n_splits > len(y) // 4 and len(y) // 4 > 1
        ):  # Ensure at least 4 data points per split if possible
            n_splits = max(2, len(y) // 4)

    # Ensure strictly increasing cumulative numbers
    y = np.maximum.accumulate(y)

    # Calculate minimum split size to ensure enough data for training
    # n_splits must be less than number of samples for TimeSeriesSplit
    if n_splits >= len(X):
        print(
            f"Warning: n_splits ({n_splits}) is >= number of samples ({len(X)}). Adjusting n_splits."
        )
        n_splits = max(
            2, len(X) - 1
        )  # Ensure n_splits is at least 2 if possible, and less than n_samples

    min_samples_for_split = len(y) // (n_splits + 1)  # Approx samples in first train split

    # Ensure min_samples_for_split is reasonable, e.g., at least a few weeks
    min_required_data_per_split = 4  # e.g. 4 weeks
    if min_samples_for_split < min_required_data_per_split and n_splits > 2:
        # Adjust n_splits to increase samples per split
        n_splits = max(2, len(y) // min_required_data_per_split - 1)
        print(
            f"Adjusted n_splits to {n_splits} to ensure at least {min_required_data_per_split} samples per split."
        )
        if n_splits >= len(X):  # Re-check after adjustment
            n_splits = max(2, len(X) - 1)

    # Create TimeSeriesSplit object with gap
    # Ensure n_splits is valid (must be > 1 for TimeSeriesSplit if used for CV)
    if n_splits < 2:
        print(
            f"Warning: n_splits is {n_splits}. TimeSeriesSplit requires n_splits >= 2. Setting to 2."
        )
        n_splits = 2
        # If len(X) is also very small, this could still be an issue.
        if n_splits >= len(X):  # e.g. if len(X) is 2, n_splits becomes 1
            if len(X) > 1:
                n_splits = len(X) - 1
            else:  # len(X) is 1 or 0
                raise ValueError(f"Cannot perform time series split with {len(X)} samples.")

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=4)  # 4-week gap between train and test

    # Store all splits for cross-validation
    try:
        cv_splits = list(tscv.split(X))
    except ValueError as e:
        raise ValueError(
            f"Error during TimeSeriesSplit: {e}. n_splits={n_splits}, X length={len(X)}"
        )

    print(f"\nPrepared {n_splits} time series splits for cross-validation.")
    # Recalculate min_samples based on final n_splits
    final_min_samples = len(y) // (n_splits + 1) if n_splits > 0 else len(y)
    print(f"Approx data points in first training split: ~{final_min_samples}")
    print(f"Debug: Length of X: {len(X)}, Length of y: {len(y)}")

    return X, y, cv_splits, tscv
