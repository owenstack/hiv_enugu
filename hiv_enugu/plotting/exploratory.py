import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
from .utils import plot_manager


@plot_manager
def plot_exploratory_visualizations(df, **kwargs):
    """Generate and save a grid of exploratory data visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle("Exploratory Data Analysis of HIV Cases in Enugu State", fontsize=20)

    # Plot 1: Time Series of New Enrollments
    ax = axes[0, 0]
    ax.plot(df['date'], df['enrollment'], label='Newly Enrolled', color='teal')
    ax.set_title('Time Series of New HIV Enrollments')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of People')
    ax.legend()

    # Plot 2: Yearly Distribution of New Enrollments
    ax = axes[0, 1]
    sns.boxplot(x='year', y='enrollment', data=df, ax=ax)
    ax.set_title('Yearly Distribution of New Enrollments')
    ax.set_xlabel('Year')
    ax.set_ylabel('Newly Enrolled')
    ax.tick_params(axis='x', rotation=45)

    # Plot 3: Monthly Trend of New Enrollments
    ax = axes[1, 0]
    monthly_trend = df.groupby('month')['enrollment'].mean()
    monthly_trend.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Average Monthly Trend of New Enrollments')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Number of People')
    ax.tick_params(axis='x', rotation=0)

    # Plot 4: Cumulative Enrollments Over Time
    ax = axes[1, 1]
    ax.plot(df['date'], df['cumulative'], label='Cumulative', color='purple')
    ax.set_title('Cumulative HIV Cases Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Number of People')
    ax.legend()

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig