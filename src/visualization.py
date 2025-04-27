from typing import Optional, List
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import log_info, log_warning, log_error

def create_trend_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: str,
    hue: Optional[str] = None,
    style: str = 'line'
) -> None:
    """
    Create and save a trend plot.

    Args:
        df (pd.DataFrame): Input dataframe.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        title (str): Plot title.
        output_path (str): Path to save the plot.
        hue (Optional[str]): Column name for color grouping.
        style (str): Plot style ('line' or 'scatter').
    """
    log_info(f"Creating trend plot: {title}")
    
    plt.figure(figsize=(12, 6))
    
    try:
        if style == 'line':
            if hue:
                sns.lineplot(data=df, x=x_col, y=y_col, hue=hue)
            else:
                sns.lineplot(data=df, x=x_col, y=y_col)
        elif style == 'scatter':
            if hue:
                sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue)
            else:
                sns.scatterplot(data=df, x=x_col, y=y_col)
        else:
            log_warning(f"Unsupported plot style: {style}. Defaulting to line plot.")
            sns.lineplot(data=df, x=x_col, y=y_col)
        
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.savefig(output_path)
        plt.close()
        
        log_info(f"Plot saved to {output_path}")
        
    except Exception as e:
        log_error(f"Error creating trend plot: {str(e)}")
        plt.close()
        raise

def create_correlation_heatmap(
    df: pd.DataFrame,
    columns: List[str],
    title: str,
    output_path: str
) -> None:
    """
    Create and save a correlation heatmap.

    Args:
        df (pd.DataFrame): Input dataframe.
        columns (List[str]): List of columns to include in correlation.
        title (str): Plot title.
        output_path (str): Path to save the plot.
    """
    log_info(f"Creating correlation heatmap: {title}")
    
    try:
        # Calculate correlation matrix
        corr_matrix = df[columns].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        
        plt.title(title)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.savefig(output_path)
        plt.close()
        
        log_info(f"Correlation heatmap saved to {output_path}")
        
    except Exception as e:
        log_error(f"Error creating correlation heatmap: {str(e)}")
        plt.close()
        raise

def create_distribution_plot(
    df: pd.DataFrame,
    column: str,
    title: str,
    output_path: str,
    bins: int = 30
) -> None:
    """
    Create and save a distribution plot.

    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column to plot distribution for.
        title (str): Plot title.
        output_path (str): Path to save the plot.
        bins (int): Number of bins for histogram.
    """
    log_info(f"Creating distribution plot for column: {column}")
    
    try:
        plt.figure(figsize=(10, 6))
        
        sns.histplot(data=df, x=column, bins=bins)
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel('Count')
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.savefig(output_path)
        plt.close()
        
        log_info(f"Distribution plot saved to {output_path}")
        
    except Exception as e:
        log_error(f"Error creating distribution plot: {str(e)}")
        plt.close()
        raise 