"""
Configuration settings for AI Media Trends Insight Analyzer.
Contains project paths, model parameters, and other constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Output directories
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PLOTS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Logging configuration
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

# Model parameters
MODEL_PARAMS = {
    "random_state": 42,
    "test_size": 0.2,
    "n_splits": 5
}

# Data processing parameters
DATA_PROCESSING = {
    "date_format": "%Y-%m-%d",
    "min_samples": 100,
    "outlier_threshold": 3
}

# Visualization settings
PLOT_SETTINGS = {
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "seaborn",
    "palette": "deep"
}

# API configuration
API_CONFIG = {
    "base_url": os.getenv("API_BASE_URL", "http://localhost:8000"),
    "timeout": 30,
    "max_retries": 3
}

# LLM configuration
LLM_CONFIG = {
    "model_name": "llama-2-7b",
    "temperature": 0.7,
    "max_length": 512,
    "top_p": 0.95
}

# Streamlit app settings
STREAMLIT_CONFIG = {
    "page_title": "AI Media Trends Insight Analyzer",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# API Settings
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Model Settings
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama2-7b")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 1000))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))

# Path Settings
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = os.getenv("DATA_PATH", ROOT_DIR / "data")
OUTPUT_DIR = os.getenv("OUTPUT_PATH", ROOT_DIR / "outputs")
STREAMLIT_THEME = os.getenv("STREAMLIT_THEME", "light")
PAGE_TITLE = os.getenv("PAGE_TITLE", "AI Media Trends Insight")
PAGE_ICON = os.getenv("PAGE_ICON", "📊")

# Logging Settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5
LOG_CONSOLE_FORMAT = "%(levelname)s: %(message)s"

# Ensure directories exist
for dir_path in [DATA_DIR, OUTPUT_DIR, PLOTS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True) 