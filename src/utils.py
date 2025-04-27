"""
Utility functions for AI Media Trends Insight Analyzer.
Contains helper functions for data processing, file handling, and other common operations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pandas import DataFrame

from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from .logger import logger

def load_data(
    filename: str,
    data_dir: str = RAW_DATA_DIR,
    **kwargs
) -> DataFrame:
    """
    Load data from a file into a pandas DataFrame.
    
    Args:
        filename: Name of the file to load
        data_dir: Directory containing the file (default: RAW_DATA_DIR)
        **kwargs: Additional arguments passed to pd.read_csv/read_parquet
    
    Returns:
        DataFrame: Loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    file_path = Path(data_dir) / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading data from {file_path}")
    
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path, **kwargs)
    elif file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Loaded DataFrame with shape {df.shape}")
    return df

def save_data(
    df: DataFrame,
    filename: str,
    data_dir: str = PROCESSED_DATA_DIR,
    **kwargs
) -> None:
    """
    Save DataFrame to a file.
    
    Args:
        df: DataFrame to save
        filename: Output filename
        data_dir: Output directory (default: PROCESSED_DATA_DIR)
        **kwargs: Additional arguments passed to df.to_csv/to_parquet
    """
    output_path = Path(data_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving DataFrame with shape {df.shape} to {output_path}")
    
    if output_path.suffix == '.csv':
        df.to_csv(output_path, index=False, **kwargs)
    elif output_path.suffix == '.parquet':
        df.to_parquet(output_path, index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {output_path.suffix}")

def validate_dataframe(
    df: DataFrame,
    required_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None
) -> bool:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
        numeric_columns: List of columns that must be numeric
    
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if numeric_columns:
        non_numeric = [col for col in numeric_columns 
                      if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            raise ValueError(f"Non-numeric columns that should be numeric: {non_numeric}")
    
    return True

def save_json(
    data: Union[Dict, List],
    filename: str,
    output_dir: str = PROCESSED_DATA_DIR
) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save (must be JSON serializable)
        filename: Output filename
        output_dir: Output directory
    """
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving JSON data to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(
    filename: str,
    data_dir: str = RAW_DATA_DIR
) -> Any:
    """
    Load data from JSON file.
    
    Args:
        filename: Name of the file to load
        data_dir: Directory containing the file
    
    Returns:
        Loaded JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(data_dir) / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading JSON data from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_timestamp() -> str:
    """
    Get current timestamp in YYYYMMDD_HHMMSS format.
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S") 