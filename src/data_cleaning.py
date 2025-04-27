import os
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import log_info, log_warning

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    log_info(f"Loading dataset from {filepath}")
    return pd.read_csv(filepath)

def display_data_overview(df: pd.DataFrame) -> None:
    """
    Display the first 5 rows, info, and statistical summary of the dataset.

    Args:
        df (pd.DataFrame): The dataset.
    """
    log_info("Displaying first 5 rows of the dataset:")
    print(df.head())
    log_info("Dataset info:")
    print(df.info())
    log_info("Statistical summary:")
    print(df.describe(include='all'))

def analyze_missing_values(df: pd.DataFrame) -> None:
    """
    Print the number of missing values per column.

    Args:
        df (pd.DataFrame): The dataset.
    """
    missing = df.isnull().sum()
    log_info("Missing values per column:")
    print(missing)

def plot_histogram_ai_usage(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create and save a histogram of AI Usage % over the years.

    Args:
        df (pd.DataFrame): The dataset.
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='AI Adoption Rate (%)', bins=20, kde=True)
    plt.title('Distribution of AI Adoption Rate (%)')
    plt.xlabel('AI Adoption Rate (%)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'ai_usage_percent_histogram.png')
    plt.savefig(plot_path)
    plt.close()
    log_info(f"Saved histogram to {plot_path}")

def plot_bar_category_counts(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create and save a bar plot of the number of records per Industry.

    Args:
        df (pd.DataFrame): The dataset.
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Industry', order=df['Industry'].value_counts().index)
    plt.title('Number of Records per Industry')
    plt.xlabel('Industry')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'industry_counts_barplot.png')
    plt.savefig(plot_path)
    plt.close()
    log_info(f"Saved bar plot to {plot_path}")

def plot_line_avg_ai_usage_by_category(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create and save a line chart of average AI Usage % per Industry over time.

    Args:
        df (pd.DataFrame): The dataset.
        output_dir (str): Directory to save the plot.
    """
    if 'Year' not in df.columns:
        log_warning("Column 'Year' not found in dataset. Skipping line plot.")
        return
    plt.figure(figsize=(12, 7))
    avg_usage = df.groupby(['Year', 'Industry'])['AI Adoption Rate (%)'].mean().reset_index()
    sns.lineplot(data=avg_usage, x='Year', y='AI Adoption Rate (%)', hue='Industry', marker='o')
    plt.title('Average AI Adoption Rate (%) per Industry Over Time')
    plt.xlabel('Year')
    plt.ylabel('Average AI Adoption Rate (%)')
    plt.legend(title='Industry', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'avg_ai_usage_by_industry_lineplot.png')
    plt.savefig(plot_path)
    plt.close()
    log_info(f"Saved line plot to {plot_path}")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values.
    Drops rows with critical missing information (Year, Industry, AI Usage %).
    Imputes or flags other missing values as needed.

    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    # Drop rows with missing critical fields
    cleaned_df = df.dropna(subset=['Year', 'Industry', 'AI Adoption Rate (%)'])
    # Example: fill missing values in non-critical columns if needed
    # cleaned_df['SomeColumn'] = cleaned_df['SomeColumn'].fillna('Unknown')
    log_info(f"Dropped {len(df) - len(cleaned_df)} rows with critical missing values.")
    return cleaned_df

def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the cleaned dataset to a CSV file.

    Args:
        df (pd.DataFrame): The cleaned dataset.
        output_path (str): Path to save the CSV file.
    """
    df.to_csv(output_path, index=False)
    log_info(f"Cleaned data saved to {output_path}")

def main() -> None:
    """
    Main function to orchestrate the data exploration and cleaning workflow.
    """
    raw_data_path = os.path.join('data', 'raw', 'Global_AI_Content_Impact_Dataset.csv')
    processed_data_path = os.path.join('data', 'processed', 'ai_impact_data_cleaned.csv')
    plots_dir = os.path.join('outputs', 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    df = load_dataset(raw_data_path)
    display_data_overview(df)
    analyze_missing_values(df)
    plot_histogram_ai_usage(df, plots_dir)
    plot_bar_category_counts(df, plots_dir)
    plot_line_avg_ai_usage_by_category(df, plots_dir)
    cleaned_df = clean_data(df)
    save_cleaned_data(cleaned_df, processed_data_path)

if __name__ == "__main__":
    main()
