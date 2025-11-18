"""
Performance metrics, contribution analysis, and ROI/ROAS calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def compute_fit_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute model fit metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary with R², RMSE, and MAPE
    """
    residuals = y_true - y_pred
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    
    # MAPE (Mean Absolute Percentage Error)
    mask = y_true != 0
    mape = np.mean(np.abs(residuals[mask] / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
    
    return {
        "R2": r2,
        "RMSE": rmse,
        "MAPE": mape
    }


def compute_contributions(
    X_saturated: np.ndarray,
    beta_means: np.ndarray,
    alpha_mean: float,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    media_cols: list
) -> Tuple[pd.DataFrame, float, float]:
    """
    Compute channel contributions in original scale.
    
    Transforms standardized coefficients back to original scale and
    calculates each channel's contribution to total sales.
    
    Args:
        X_saturated: Saturated media features in original scale (n_samples, n_features)
        beta_means: Mean beta coefficients from posterior (n_features,)
        alpha_mean: Mean intercept from posterior
        scaler_X: Fitted StandardScaler for features
        scaler_y: Fitted StandardScaler for target
        media_cols: List of media column names
        
    Returns:
        Tuple of (contributions_df, baseline_total, residual_total)
    """
    n_samples = X_saturated.shape[0]
    
    # Transform coefficients to original scale
    sigma_y = scaler_y.scale_[0]
    mu_y = scaler_y.mean_[0]
    sigma_x = scaler_X.scale_
    mu_x = scaler_X.mean_
    
    # Coefficients in original scale: gamma = sigma_y * beta / sigma_x
    gamma = sigma_y * beta_means / sigma_x
    
    # Baseline in original scale
    const = mu_y + sigma_y * alpha_mean - np.sum(gamma * mu_x)
    baseline_total = const * n_samples
    
    # Channel contributions
    contributions = {}
    for j, channel in enumerate(media_cols):
        Xj = X_saturated[:, j]
        contributions[channel] = float(gamma[j] * Xj.sum())
    
    # Create DataFrame
    contrib_df = pd.DataFrame({
        "Canal": media_cols,
        "Coeficiente_gamma": gamma,
        "Contribución_total": [contributions[ch] for ch in media_cols]
    })
    
    return contrib_df, baseline_total, contributions


def compute_roi_roas(
    contrib_df: pd.DataFrame,
    df_original: pd.DataFrame,
    media_cols: list,
    total_sales: float
) -> pd.DataFrame:
    """
    Compute ROI and ROAS (corrected definitions) for each channel.
    
    ROI = (Contribution - Investment) / Investment
    ROAS = Contribution / Investment (Revenue per dollar spent)
    Share of Sales = Contribution / Total Sales
    
    Args:
        contrib_df: DataFrame with contributions per channel
        df_original: Original DataFrame with spending data
        media_cols: List of media column names
        total_sales: Total sales value
        
    Returns:
        Updated DataFrame with ROI, ROAS, and investment columns
    """
    investments = []
    roi_list = []
    roas_list = []
    share_of_sales_list = []
    
    for channel in media_cols:
        # Total investment (spending) for this channel
        investment = float(df_original[channel].sum())
        investments.append(investment)
        
        # Get contribution for this channel
        contrib = contrib_df.loc[contrib_df["Canal"] == channel, "Contribución_total"].values[0]
        
        # ROI: (Return - Investment) / Investment
        roi = (contrib - investment) / investment if investment != 0 else np.nan
        roi_list.append(roi)
        
        # ROAS: Return / Investment (revenue per dollar spent)
        roas = contrib / investment if investment != 0 else np.nan
        roas_list.append(roas)
        
        # Share of total sales
        share = contrib / total_sales if total_sales != 0 else np.nan
        share_of_sales_list.append(share)
    
    # Add columns to DataFrame
    contrib_df["Inversión_total"] = investments
    contrib_df["ROI"] = roi_list
    contrib_df["ROAS"] = roas_list
    contrib_df["Share_of_Sales"] = share_of_sales_list
    
    return contrib_df


def compute_residual(
    y_true: np.ndarray,
    baseline_total: float,
    contributions: Dict[str, float]
) -> float:
    """
    Compute residual between actual sales and model components.
    
    Args:
        y_true: Actual sales values
        baseline_total: Total baseline contribution
        contributions: Dictionary of channel contributions
        
    Returns:
        Residual value (should be close to 0 for good fit)
    """
    total_sales = float(y_true.sum())
    total_explained = baseline_total + sum(contributions.values())
    residual = total_sales - total_explained
    
    # Clean up very small residuals (numerical precision)
    if abs(residual) < 1e-6 * max(total_sales, 1.0):
        residual = 0.0
    
    return residual


def format_metrics_display(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Format metrics for display in UI.
    
    Args:
        metrics: Dictionary with metric values
        
    Returns:
        Dictionary with formatted string values
    """
    formatted = {}
    
    if "R2" in metrics:
        formatted["R²"] = f"{metrics['R2']:.3f}" if not np.isnan(metrics['R2']) else "N/A"
    
    if "RMSE" in metrics:
        formatted["RMSE"] = f"{metrics['RMSE']:,.3f}"
    
    if "MAPE" in metrics:
        formatted["MAPE"] = f"{metrics['MAPE']:.2f}%" if not np.isnan(metrics['MAPE']) else "N/A"
    
    return formatted
