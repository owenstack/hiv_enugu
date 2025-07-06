import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from hiv_enugu.config import PROCESSED_DATA_DIR
from pandas import DataFrame


def load_and_prepare_data_for_modeling(path=PROCESSED_DATA_DIR / "cleaned_enrollments.csv"):
    """Loads and prepares the data for modeling."""
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        df["cumulative"] = df["enrollment_cleaned"].cumsum()
        df["time_idx"] = (df["date"] - df["date"].min()).dt.days
        return df
    except FileNotFoundError:
        print(f"Error: The file {path} was not found.")
        return None


def get_cv_splits(df: DataFrame, n_splits=5):
    """Creates cross-validation splits for time series data."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(df))
