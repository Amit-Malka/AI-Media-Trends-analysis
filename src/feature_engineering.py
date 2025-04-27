import os
import logging
from typing import Optional
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_dataset() -> Optional[pd.DataFrame]:
    """
    Load the cleaned dataset from processed or raw directory.

    Returns:
        Optional[pd.DataFrame]: Loaded DataFrame or None if not found.
    """
    processed_path = os.path.join('data', 'processed', 'Global_AI_Content_Impact_Dataset.csv')
    raw_path = os.path.join('data', 'raw', 'Global_AI_Content_Impact_Dataset.csv')
    try:
        if os.path.exists(processed_path):
            logging.info(f"Loading dataset from {processed_path}")
            return pd.read_csv(processed_path)
        elif os.path.exists(raw_path):
            logging.info(f"Loading dataset from {raw_path}")
            return pd.read_csv(raw_path)
        else:
            logging.error("Dataset file not found in processed or raw directory.")
            return None
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return None

def add_yearly_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column for yearly percentage change of AI Adoption Rate (%) per Industry.

    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: DataFrame with new column 'yearly_change_%'.
    """
    df = df.sort_values(['Industry', 'Year'])
    df['yearly_change_%'] = df.groupby('Industry')['AI Adoption Rate (%)'].pct_change() * 100
    return df

def add_growing_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean column indicating if AI Adoption Rate (%) has been increasing for at least 3 consecutive years per Industry.

    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: DataFrame with new column 'is_growing_trend'.
    """
    def is_growing(group: pd.Series) -> pd.Series:
        trend = group > group.shift(1)
        rolling = trend.rolling(window=3, min_periods=3).sum() == 3
        return rolling.fillna(False)
    df['is_growing_trend'] = df.groupby('Industry')['AI Adoption Rate (%)'].apply(is_growing).reset_index(level=0, drop=True)
    return df

def add_impact_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column 'impact_summary' combining Industry, Year, AI Adoption Rate (%), and a description.

    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: DataFrame with new column 'impact_summary'.
    """
    def make_summary(row):
        return (
            f"Industry: {row.get('Industry', 'N/A')}, Year: {row.get('Year', 'N/A')}, "
            f"AI Adoption Rate: {row.get('AI Adoption Rate (%)', 'N/A')}%, "
            f"Description: {row.get('AI Impact Description', 'N/A')}"
        )
    df['impact_summary'] = df.apply(make_summary, axis=1)
    return df

def generate_text_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a text profile for each Industry-Year combination and add as 'text_profile'.

    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: DataFrame with new column 'text_profile'.
    """
    def get_trend(row):
        if pd.isna(row.get('is_growing_trend')):
            return 'unknown'
        return 'growing' if row['is_growing_trend'] else 'declining'
    def safe_val(val):
        return 'N/A' if pd.isna(val) else val
    def make_profile(row):
        return (
            f"In {safe_val(row.get('Year'))}, within the {safe_val(row.get('Industry'))} sector, "
            f"the AI usage reached {safe_val(row.get('AI Adoption Rate (%)'))}%. "
            f"The overall impact of AI is described as: '{safe_val(row.get('AI Impact Description'))}'. "
            f"Compared to previous years, this represents a {safe_val(row.get('yearly_change_%'))}% change. "
            f"The trend is currently {get_trend(row)}."
        )
    df['text_profile'] = df.apply(make_profile, axis=1)
    return df

def save_outputs(df: pd.DataFrame) -> None:
    """
    Save the engineered dataset and text profiles.

    Args:
        df (pd.DataFrame): The dataset with engineered features.
    """
    processed_path = os.path.join('data', 'processed', 'Global_AI_Content_Impact_Dataset_Engineered.csv')
    profiles_path = os.path.join('outputs', 'reports', 'text_profiles.txt')
    try:
        df.to_csv(processed_path, index=False)
        logging.info(f"Engineered dataset saved to {processed_path}")
    except Exception as e:
        logging.error(f"Error saving engineered dataset: {e}")
    try:
        with open(profiles_path, 'w', encoding='utf-8') as f:
            for profile in df['text_profile']:
                f.write(str(profile) + '\n')
        logging.info(f"Text profiles saved to {profiles_path}")
    except Exception as e:
        logging.error(f"Error saving text profiles: {e}")

def main() -> None:
    """
    Main function to orchestrate feature engineering and text profile generation.
    """
    df = load_dataset()
    if df is None:
        logging.error("No data to process. Exiting.")
        return
    # Ensure 'AI Impact Description' exists (if not, create dummy)
    if 'AI Impact Description' not in df.columns:
        df['AI Impact Description'] = 'No description available.'
    df = add_yearly_change(df)
    df = add_growing_trend(df)
    df = add_impact_summary(df)
    df = generate_text_profiles(df)
    save_outputs(df)

if __name__ == "__main__":
    main()
