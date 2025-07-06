import pandas as pd
import numpy as np
import os
from plotting.exploratory import (
    plot_yearly_enrollment_trends,
    plot_yearly_distribution,
    plot_monthly_enrollment_trends,
    plot_enrollment_boxplot,
    plot_enrollment_timeseries,
    plot_cleaned_timeseries,
)
from hiv_enugu.config import FIGURES_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from loguru import logger
from pathlib import Path
from pandas import DataFrame


def load_and_clean_data(path: Path):
    """Loads data from Excel, handles duplicates, and creates a complete date range."""
    logger.info("1. Loading and cleaning data from {path}")
    logger.info("-" * 50)
    if not os.path.exists(path):
        logger.error(f"Error: The file {path} was not found.")
        return None

    df = pd.read_excel(path)
    df = df[["date", "enrollment"]]
    df["date"] = pd.to_datetime(df["date"])

    # Handle duplicates before reindexing
    if df["date"].duplicated().any():
        logger.info("Duplicate dates found. Aggregating by summing enrollments.")
        df = df.groupby("date")["enrollment"].sum().reset_index()

    # Create a complete date range
    full_date_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")
    df_complete = (
        df.set_index("date")
        .reindex(full_date_range)
        .reset_index()
        .rename(columns={"index": "date"})
    )

    logger.info(
        f"Data loaded and cleaned. Date range from {df_complete['date'].min().date()} to {df_complete['date'].max().date()}."
    )
    return df_complete


def analyze_and_visualize_trends(df: DataFrame):
    """Performs trend analysis and generates visualizations."""
    logger.info("\n2. ANALYZING AND VISUALIZING TRENDS")
    logger.info("-" * 50)
    df["Year"] = df["date"].dt.year
    df["Month"] = df["date"].dt.month_name()
    df["DayOfWeek"] = df["date"].dt.day_name()

    # Yearly analysis
    yearly_stats = df.groupby("Year")["enrollment"].agg(["mean", "std", "count"])
    yearly_stats.columns = ["Mean", "Std", "Count"]
    logger.info("\nYearly Statistics:")
    logger.info(yearly_stats)

    # Monthly analysis
    monthly_stats = df.groupby("Month")["enrollment"].agg(["mean", "std", "count"])
    logger.info("\nMonthly Statistics:")
    logger.info(monthly_stats)

    # Generate plots
    plot_yearly_enrollment_trends(yearly_stats, filename="yearly_trends.png")
    plot_yearly_distribution(df, filename="yearly_distributions.png")
    plot_monthly_enrollment_trends(monthly_stats, filename="monthly_trends.png")
    logger.info("\nTrend visualizations saved.")


def analyze_outliers(df: DataFrame):
    """Identifies and analyzes outliers in the enrollment data."""
    logger.info("\n3. ANALYZING OUTLIERS")
    logger.info("-" * 50)
    Q1 = df["enrollment"].quantile(0.25)
    Q3 = df["enrollment"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df["enrollment"] < lower_bound) | (df["enrollment"] > upper_bound)]
    logger.info(f"Found {len(outliers)} outliers.")
    if not outliers.empty:
        logger.info("Top 5 outliers:")
        logger.info(outliers.nlargest(5, "enrollment"))

    # Generate plots
    plot_enrollment_boxplot(df, filename="enrollments_boxplot.png")
    plot_enrollment_timeseries(df, filename="enrollments_timeseries.png")
    logger.info("\nOutlier visualizations saved.")


def finalize_and_save(df: DataFrame, output_path: Path):
    """Handles missing values, performs final checks, and saves the cleaned data."""
    logger.info("\n4. FINALIZING AND SAVING DATA")
    logger.info("-" * 50)
    # Data validity check
    logger.info(f"Number of negative values: {(df['enrollment'] < 0).sum()}")
    non_integer_mask = df["enrollment"].notna() & (df["enrollment"] % 1 != 0)
    logger.info(f"Number of non-integer values: {non_integer_mask.sum()}")

    # Handle missing values by filling with zero
    df["enrollment_cleaned"] = df["enrollment"].fillna(0)
    logger.info(f"\nOriginal NaN count: {df['enrollment'].isna().sum()}")
    logger.info(f"Cleaned NaN count: {df['enrollment_cleaned'].isna().sum()}")

    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"\nAnalysis complete. Cleaned data saved to {output_path}")

    # Plot cleaned data
    plot_cleaned_timeseries(df, filename="enrollments_cleaned_timeseries.png")
    logger.info("Cleaned data visualization saved.")


def main():
    """Main function to run the full data analysis pipeline."""
    # Ensure figures directory exists
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df_complete = load_and_clean_data(RAW_DATA_DIR / "data.xlsx")
    if df_complete is not None:
        analyze_and_visualize_trends(df_complete)
        analyze_outliers(df_complete)
        finalize_and_save(df_complete, PROCESSED_DATA_DIR / "cleaned_enrollments.csv")


if __name__ == "__main__":
    main()
