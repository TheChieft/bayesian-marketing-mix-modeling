"""
Transformation functions for Marketing Mix Modeling.

Includes adstock, saturation (Hill), and standardization utilities.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler


def adstock(x: np.ndarray, rate: float = 0.1) -> np.ndarray:
    """
    Apply adstock transformation with decay rate.
    
    The adstock effect models carryover: 
        y[t] = x[t] + rate * y[t-1]
    
    Args:
        x: Input array (can be pd.Series or np.ndarray)
        rate: Decay rate (0 to 1), how much of previous effect carries over
        
    Returns:
        Transformed array with adstock effect applied
    """
    if isinstance(x, pd.Series):
        x = x.values
    
    output = np.zeros(len(x))
    output[0] = x[0]
    
    for t in range(1, len(x)):
        output[t] = x[t] + rate * output[t - 1]
    
    return output


def hill(x: np.ndarray, alpha: float = 1.0, theta: float = 1.0, gamma: float = 1.5) -> np.ndarray:
    """
    Apply Hill saturation transformation.
    
    Models diminishing returns with the formula:
        f(x) = alpha * x^gamma / (theta^gamma + x^gamma)
    
    Args:
        x: Input array
        alpha: Maximum response (saturation level)
        theta: Half-saturation point (x value at 50% of alpha)
        gamma: Shape parameter (curve steepness, > 1 is S-shaped)
        
    Returns:
        Saturated values
    """
    x = np.array(x, dtype=float)
    return alpha * (x ** gamma) / (theta ** gamma + x ** gamma)


def build_transformed_media(
    df: pd.DataFrame,
    media_cols: List[str],
    adstock_rate: float,
    hill_gamma: float,
    hill_alpha: float = 1.0
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply adstock and Hill saturation to all media columns.
    
    Creates new columns:
        - {col}_ad: After adstock transformation
        - {col}_sat: After saturation transformation
    
    Args:
        df: DataFrame with media spending columns
        media_cols: List of column names to transform
        adstock_rate: Decay rate for adstock
        hill_gamma: Shape parameter for Hill function
        hill_alpha: Maximum response for Hill function
        
    Returns:
        Tuple of (transformed DataFrame, list of saturated column names)
    """
    df_trans = df.copy()
    saturated_cols = []
    
    for col in media_cols:
        # Apply adstock
        ad_col = f"{col}_ad"
        df_trans[ad_col] = adstock(df[col].values, rate=adstock_rate)
        
        # Calculate theta as mean of adstocked values (or 1.0 if mean is 0)
        theta = df_trans[ad_col].mean() if df_trans[ad_col].mean() > 0 else 1.0
        
        # Apply Hill saturation
        sat_col = f"{col}_sat"
        df_trans[sat_col] = hill(
            df_trans[ad_col].values,
            alpha=hill_alpha,
            theta=theta,
            gamma=hill_gamma
        )
        
        saturated_cols.append(sat_col)
    
    return df_trans, saturated_cols


def standardize_data(
    X: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """
    Standardize features and target using StandardScaler.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        
    Returns:
        Tuple of (X_scaled, y_scaled, scaler_X, scaler_y)
    """
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X.astype('float32'))
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    return X_scaled.astype('float32'), y_scaled.astype('float32'), scaler_X, scaler_y


def inverse_transform_predictions(
    y_pred_scaled: np.ndarray,
    scaler_y: StandardScaler
) -> np.ndarray:
    """
    Transform predictions back to original scale.
    
    Args:
        y_pred_scaled: Predictions in standardized scale
        scaler_y: Fitted StandardScaler for target variable
        
    Returns:
        Predictions in original scale
    """
    return y_pred_scaled * scaler_y.scale_[0] + scaler_y.mean_[0]
