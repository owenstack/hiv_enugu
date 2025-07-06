import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
from .utils import plot_manager
from pandas import DataFrame


@plot_manager
def plot_yearly_enrollment_trends(yearly_stats: DataFrame, **kwargs):
    """Plots the yearly trends for mean enrollments and zero enrollment days."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart for mean enrollments
    ax1.bar(
        yearly_stats.index, yearly_stats["Mean"], color="skyblue", label="Mean Daily Enrollment"
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Mean Daily Enrollment", color="skyblue")
    ax1.tick_params(axis="y", labelcolor="skyblue")

    # Line chart for standard deviation
    ax2 = ax1.twinx()
    ax2.plot(
        yearly_stats.index,
        yearly_stats["Std"],
        color="coral",
        marker="o",
        label="Std. Dev. of Daily Enrollment",
    )
    ax2.set_ylabel("Standard Deviation", color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")

    plt.title("Yearly Enrollment Trends")
    fig.tight_layout()
    return fig


@plot_manager
def plot_yearly_distribution(df: DataFrame, **kwargs):
    """Plots the distribution of enrollments by year."""
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(data=df, x="Year", y="enrollment", ax=ax)
    plt.title("Enrollment Distribution by Year")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


@plot_manager
def plot_monthly_enrollment_trends(monthly_stats: DataFrame, **kwargs):
    """Plots the average daily enrollments by month."""
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    monthly_stats.index = pd.CategoricalIndex(
        monthly_stats.index, categories=month_order, ordered=True
    )
    monthly_stats = monthly_stats.sort_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(monthly_stats.index, monthly_stats["mean"])
    plt.title("Average Enrollments by Month")
    plt.xlabel("Month")
    plt.ylabel("Mean Daily Enrollments")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


@plot_manager
def plot_enrollment_boxplot(df: DataFrame, **kwargs):
    """Plots a box plot of daily enrollments."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(y=df["enrollment"], ax=ax)
    plt.title("Box Plot of Daily Enrollments")
    plt.ylabel("Number of Enrollments")
    plt.tight_layout()
    return fig


@plot_manager
def plot_enrollment_timeseries(df: DataFrame, **kwargs):
    """Plots the time series of daily enrollments."""
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df["date"], df["enrollment"])
    plt.title("Time Series of Daily Enrollments")
    plt.xlabel("Date")
    plt.ylabel("Number of Enrollments")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


@plot_manager
def plot_cleaned_timeseries(df: DataFrame, **kwargs):
    """Plots the time series of cleaned daily enrollments."""
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df["date"], df["enrollment_cleaned"])
    plt.title("Time Series of Cleaned Daily Enrollments")
    plt.xlabel("Date")
    plt.ylabel("Number of Enrollments")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
