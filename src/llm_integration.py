import os
import logging
from typing import List
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'Global_AI_Content_Impact_Dataset_with_Insights.csv')

def load_profiles(csv_path: str) -> pd.DataFrame:
    """
    Load the engineered dataset and return the DataFrame.

    Args:
        csv_path (str): Path to the engineered CSV file.

    Returns:
        pd.DataFrame: DataFrame with text profiles.
    """
    try:
        df = pd.read_csv(csv_path)
        if 'text_profile' not in df.columns:
            raise ValueError("'text_profile' column not found in dataset.")
        return df
    except Exception as e:
        logging.error(f"Error loading profiles: {e}")
        raise

def load_llm_model(model_name: str = "google/flan-t5-base", device: str = "cpu"):
    """
    Load a Huggingface LLM model and tokenizer for summarization/instruction following.

    Args:
        model_name (str): Model name from Huggingface hub.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').

    Returns:
        pipeline: Huggingface pipeline for text2text generation.
    """
    try:
        summarizer = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            device=0 if device == "cuda" else -1,
            max_new_tokens=128
        )
        return summarizer
    except Exception as e:
        logging.error(f"Error loading LLM model: {e}")
        raise

def generate_insight(prompt: str, summarizer) -> str:
    """
    Generate a business-style insight using the LLM for a given text profile.

    Args:
        prompt (str): The text profile to summarize.
        summarizer: Huggingface pipeline.

    Returns:
        str: Generated insight.
    """
    instruction = (
        "Summarize the AI trend in this sector and year in 2-3 sentences for a business audience. "
        "Highlight growth, decline, or major changes if mentioned.\n\nProfile: "
    )
    try:
        result = summarizer(instruction + prompt)
        if isinstance(result, list) and len(result) > 0:
            return result[0]['generated_text'].strip()
        return ""
    except Exception as e:
        logging.error(f"LLM failed on profile: {e}")
        return "[LLM Error]"

def batch_generate_insights(profiles: List[str], summarizer, batch_size: int = 4) -> List[str]:
    """
    Generate insights for a list of profiles using batching.

    Args:
        profiles (List[str]): List of text profiles.
        summarizer: Huggingface pipeline.
        batch_size (int): Number of profiles per batch.

    Returns:
        List[str]: List of generated insights.
    """
    insights = []
    for i in tqdm(range(0, len(profiles), batch_size), desc="Generating LLM Insights"):
        batch = profiles[i:i+batch_size]
        for prompt in batch:
            insight = generate_insight(prompt, summarizer)
            insights.append(insight)
    return insights

def save_outputs(df: pd.DataFrame, insights: List[str]) -> None:
    """
    Save the DataFrame with insights and a text file with all insights.

    Args:
        df (pd.DataFrame): The original DataFrame.
        insights (List[str]): List of generated insights.
    """
    df['llm_insight'] = insights
    csv_path = DATA_PATH
    txt_path = os.path.join('outputs', 'reports', 'llm_insights.txt')
    try:
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved dataset with insights to {csv_path}")
    except Exception as e:
        logging.error(f"Error saving CSV: {e}")
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            for insight in insights:
                f.write(insight + '\n')
        logging.info(f"Saved all insights to {txt_path}")
    except Exception as e:
        logging.error(f"Error saving insights text file: {e}")

def main() -> None:
    """
    Main function to orchestrate LLM summarization of text profiles.
    """
    df = load_profiles(DATA_PATH)
    profiles = df['text_profile'].astype(str).tolist()
    summarizer = load_llm_model()
    insights = batch_generate_insights(profiles, summarizer, batch_size=4)
    save_outputs(df, insights)

if __name__ == "__main__":
    main()
