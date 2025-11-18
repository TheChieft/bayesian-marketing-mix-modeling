"""
Data loading and validation module for MMM.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Tuple


def sanitize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Convert column names to safe identifiers.
    
    - Replaces anything that isn't [a-zA-Z0-9_] with "_"
    - Prepends "X_" if name starts with a digit
    
    Args:
        df: DataFrame with potentially unsafe column names
        
    Returns:
        Tuple of (renamed DataFrame, mapping from original to safe names)
    """
    mapping = {}
    for c in df.columns:
        safe = re.sub(r"\W+", "_", str(c))
        if safe and safe[0].isdigit():
            safe = f"X_{safe}"
        mapping[c] = safe
    return df.rename(columns=mapping), mapping


def load_base_data(path: str) -> pd.DataFrame:
    """
    Load base CSV data and perform basic validation.
    
    Args:
        path: Path to CSV file
        
    Returns:
        DataFrame with loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data has critical issues
    """
    df = pd.read_csv(path)
    
    # Basic validation
    if df.empty:
        raise ValueError("Loaded DataFrame is empty")
    
    # Check for completely null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        raise ValueError(f"Columns with all null values: {null_cols}")
    
    return df


def generate_synthetic_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic advertising data for testing.
    
    Args:
        n: Number of observations
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic advertising data
    """
    np.random.seed(seed)
    
    tv = np.random.uniform(0, 300, n)
    radio = np.random.uniform(0, 50, n)
    news = np.random.uniform(0, 50, n)
    baseline_true = 5.0
    sales = baseline_true + 0.04 * tv + 0.3 * radio + 0.02 * news + np.random.normal(0, 1.0, n)
    
    df = pd.DataFrame({
        "Dia": np.arange(1, n + 1),
        "TV Ad Budget ($)": tv,
        "Radio Ad Budget ($)": radio,
        "Newspaper Ad Budget ($)": news,
        "Sales ($)": sales,
    })
    
    return df


def validate_columns(df: pd.DataFrame, required_cols: list) -> bool:
    """
    Validate that DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        
    Returns:
        True if all required columns present
        
    Raises:
        ValueError: If any required columns are missing
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True
