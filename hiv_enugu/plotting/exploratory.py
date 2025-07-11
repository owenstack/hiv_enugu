import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import plot_manager


@plot_manager
def plot_exploratory_visualizations(df: pd.DataFrame, **kwargs):
    """Generate and save a grid of exploratory data visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle("Exploratory Data Analysis of HIV Cases in Enugu State", fontsize=20)

    # Plot 1: Time Series of New Enrollments
    ax = axes[0, 0]
    ax.plot(df["date"], df["enrollment"], label="Newly Enrolled", color="teal")
    ax.set_title("Time Series of New HIV Enrollments")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of People")
    ax.legend()

    # Plot 2: Yearly Distribution of New Enrollments
    ax = axes[0, 1]
    sns.boxplot(x="year", y="enrollment", data=df, ax=ax)
    ax.set_title("Yearly Distribution of New Enrollments")
    ax.set_xlabel("Year")
    ax.set_ylabel("Newly Enrolled")
    ax.tick_params(axis="x", rotation=45)

    # Plot 3: Monthly Trend of New Enrollments
    ax = axes[1, 0]
    monthly_trend = df.groupby("month")["enrollment"].mean()
    monthly_trend.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Average Monthly Trend of New Enrollments")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Number of People")
    ax.tick_params(axis="x", rotation=0)

    # Plot 4: Cumulative Enrollments Over Time
    ax = axes[1, 1]
    ax.plot(df["date"], df["cumulative"], label="Cumulative", color="purple")
    ax.set_title("Cumulative HIV Cases Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Number of People")
    ax.legend()

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


@plot_manager
def plot_basic_timeseries(df: pd.DataFrame, **kwargs):  # Added **kwargs to match decorator
    """Create basic time series plot of the data for cumulative cases"""
    # df here is expected to be the weekly_df from load_data
    # It should have 'date' and 'cumulative' columns.
    fig = plt.figure(figsize=(12, 6))  # Create a new figure
    plt.plot(df["date"], df["cumulative"], marker="o", linestyle="-", alpha=0.7)
    plt.title("Cumulative HIV Cases Over Time (Weekly)")  # Adjusted title
    plt.xlabel("Date")
    plt.ylabel("Cumulative Number of HIV Patients")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig  # Return the figure object for the decorator
