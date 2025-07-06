import pandas as pd
import numpy as np
from datetime import datetime
from hiv_enugu.plotting.exploratory import plot_basic_timeseries
from pathlib import Path


def load_data(file_path: Path) -> pd.DataFrame | None:
    """Load and preprocess HIV data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")

        # Keep only relevant columns and rename
        df = df[["date", "enrollment_cleaned", "cumulative"]]
        df = df.rename(columns={"enrollment_cleaned": "number"})

        # Convert date to datetime if not already
        df["date"] = pd.to_datetime(df["date"])

        # Resample to weekly data
        df.set_index("date", inplace=True)
        weekly_df = df.resample("W").agg({"number": "sum", "cumulative": "last"}).reset_index()

        # Add rolling statistics
        weekly_df["rolling_mean_4w"] = weekly_df["number"].rolling(window=4).mean()
        weekly_df["rolling_mean_12w"] = weekly_df["number"].rolling(window=12).mean()

        # Add seasonal features
        weekly_df["month"] = weekly_df["date"].dt.month
        weekly_df["quarter"] = weekly_df["date"].dt.quarter

        # Create numerical time variable for modeling
        min_date = weekly_df["date"].min()
        weekly_df["time_idx"] = (weekly_df["date"] - min_date).dt.days / 7  # in weeks

        # Forward fill any NaN from rolling calculations, then backfill for leading NaNs
        weekly_df = weekly_df.fillna(method="ffill")  # type: ignore
        weekly_df = weekly_df.fillna(method="bfill")  # For any NaNs at the beginning after ffill

        # Basic time series plot call
        # The actual plot_basic_timeseries will be defined in the plotting module
        # and decorated with @plot_manager.
        plot_basic_timeseries(weekly_df, filename="basic_timeseries_plot.png")

        print("\nData resampled to weekly frequency with added features")
        print(f"New shape: {weekly_df.shape}")

        # Ensure time_idx is not NaN if all dates were same or only one entry after resample
        if weekly_df["time_idx"].isnull().any():
            if len(weekly_df) == 1:
                weekly_df["time_idx"] = 0  # Assign 0 if only one record
            else:
                # This case should ideally not happen if min_date is valid
                print(
                    "Warning: time_idx contains NaNs after calculation. Defaulting to row numbers."
                )
                weekly_df["time_idx"] = np.arange(len(weekly_df))

        return weekly_df

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except KeyError as e:
        print(
            f"Error: Missing expected column in CSV: {e}. Ensure 'date', 'enrollment_cleaned', 'cumulative' exist."
        )
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
