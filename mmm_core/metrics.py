"""
Performance metrics, contribution analysis, and ROI/ROAS calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
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


def scale_to_units(value: float, unit_scale: str = "original") -> float:
    """
    Scale a value to specified units for display.
    
    Args:
        value: Original value
        unit_scale: Target scale ("original", "thousands", "millions", "billions")
        
    Returns:
        Scaled value
    """
    scales = {
        "original": 1,
        "thousands": 1_000,
        "millions": 1_000_000,
        "billions": 1_000_000_000
    }
    
    return value / scales.get(unit_scale, 1)


def get_unit_label(unit_scale: str = "original", currency: str = "COP") -> str:
    """
    Get the appropriate unit label for display.
    
    Args:
        unit_scale: Scale being used
        currency: Currency code
        
    Returns:
        Formatted unit label
    """
    if unit_scale == "original":
        return currency
    elif unit_scale == "thousands":
        return f"Miles de {currency}"
    elif unit_scale == "millions":
        return f"Millones de {currency}"
    elif unit_scale == "billions":
        return f"Miles de millones de {currency}"
    else:
        return currency


def generate_business_insights(
    contrib_df: pd.DataFrame,
    total_sales: float,
    total_budget: float = None
) -> List[str]:
    """
    Generate business insights from contribution and ROI/ROAS analysis.
    
    Analyzes the contribution DataFrame to identify:
    - Top performing channels
    - Underperforming channels
    - Investment efficiency (over/under-invested channels)
    - Budget allocation recommendations
    
    Args:
        contrib_df: DataFrame with columns [Canal, Contribución_total, 
                    Inversión_total, ROI, ROAS, Share_of_Sales]
        total_sales: Total sales value
        total_budget: Total marketing budget (optional)
        
    Returns:
        List of insight strings
    """
    insights = []
    
    # Ensure required columns exist
    required_cols = ["Canal", "Contribución_total", "Inversión_total", "ROI", "ROAS", "Share_of_Sales"]
    if not all(col in contrib_df.columns for col in required_cols):
        return ["⚠️ Datos insuficientes para generar insights"]
    
    # Sort by different metrics
    by_share = contrib_df.sort_values("Share_of_Sales", ascending=False)
    by_roas = contrib_df.sort_values("ROAS", ascending=False)
    by_roi = contrib_df.sort_values("ROI", ascending=False)
    
    # Calculate budget shares
    if total_budget is None:
        total_budget = contrib_df["Inversión_total"].sum()
    
    contrib_df["Share_of_Budget"] = contrib_df["Inversión_total"] / total_budget
    
    # 1. Top performer by sales contribution
    top_channel = by_share.iloc[0]
    insights.append(
        f"🏆 **Canal de mayor impacto**: {top_channel['Canal']} genera el "
        f"{top_channel['Share_of_Sales']*100:.1f}% de las ventas totales "
        f"({top_channel['Contribución_total']:,.0f} en ventas)."
    )
    
    # 2. Best ROAS
    best_roas_channel = by_roas.iloc[0]
    if not np.isnan(best_roas_channel['ROAS']) and best_roas_channel['ROAS'] > 0:
        insights.append(
            f"💰 **Mayor eficiencia (ROAS)**: {best_roas_channel['Canal']} retorna "
            f"${best_roas_channel['ROAS']:.2f} por cada $1 invertido."
        )
    
    # 3. Best ROI
    best_roi_channel = by_roi.iloc[0]
    if not np.isnan(best_roi_channel['ROI']) and best_roi_channel['ROI'] > 0:
        insights.append(
            f"📈 **Mayor ROI**: {best_roi_channel['Canal']} con ROI de "
            f"{best_roi_channel['ROI']*100:.1f}% (ganancia neta por inversión)."
        )
    
    # 4. Underperforming channel (low ROAS + high spend)
    median_investment = contrib_df["Inversión_total"].median()
    median_roas = contrib_df["ROAS"].median()
    
    underperformers = contrib_df[
        (contrib_df["ROAS"] < median_roas) & 
        (contrib_df["Inversión_total"] > median_investment)
    ]
    
    if not underperformers.empty:
        worst = underperformers.iloc[0]
        insights.append(
            f"⚠️ **Candidato para optimización**: {worst['Canal']} tiene alto gasto "
            f"({worst['Inversión_total']:,.0f}) pero ROAS bajo ({worst['ROAS']:.2f}). "
            f"Considere reducir presupuesto o mejorar la estrategia."
        )
    
    # 5. Over/under-invested channels
    for _, row in contrib_df.iterrows():
        channel = row['Canal']
        share_budget = row['Share_of_Budget']
        share_sales = row['Share_of_Sales']
        
        # Under-invested: generates more sales than budget share
        if share_sales > share_budget * 1.2:  # 20% threshold
            diff_pct = (share_sales - share_budget) * 100
            insights.append(
                f"📊 **Sub-invertido**: {channel} genera {share_sales*100:.1f}% de ventas "
                f"con solo {share_budget*100:.1f}% del presupuesto (+{diff_pct:.1f}pp). "
                f"**Recomendación**: Aumentar inversión."
            )
        
        # Over-invested: consumes more budget than sales generation
        elif share_budget > share_sales * 1.2:  # 20% threshold
            diff_pct = (share_budget - share_sales) * 100
            insights.append(
                f"📊 **Sobre-invertido**: {channel} consume {share_budget*100:.1f}% del presupuesto "
                f"pero solo genera {share_sales*100:.1f}% de ventas (-{diff_pct:.1f}pp). "
                f"**Recomendación**: Reducir inversión o mejorar efectividad."
            )
    
    # 6. General budget efficiency
    avg_roas = contrib_df["ROAS"].mean()
    if not np.isnan(avg_roas):
        if avg_roas >= 2.0:
            insights.append(
                f"✅ **Eficiencia general**: ROAS promedio de {avg_roas:.2f} indica "
                f"excelente retorno de inversión en marketing."
            )
        elif avg_roas >= 1.0:
            insights.append(
                f"✔️ **Eficiencia aceptable**: ROAS promedio de {avg_roas:.2f} indica "
                f"retorno positivo pero con espacio para optimización."
            )
        else:
            insights.append(
                f"⚠️ **Alerta de eficiencia**: ROAS promedio de {avg_roas:.2f} sugiere "
                f"que el gasto en marketing no está generando suficiente retorno. "
                f"Se recomienda revisión estratégica."
            )
    
    # 7. Portfolio diversification
    n_channels = len(contrib_df)
    concentration = (by_share.iloc[0]["Share_of_Sales"] if n_channels > 0 else 0)
    
    if concentration > 0.6:  # Top channel > 60% of sales
        insights.append(
            f"⚠️ **Concentración de riesgo**: {by_share.iloc[0]['Canal']} representa "
            f"{concentration*100:.1f}% de las ventas. Considere diversificar canales "
            f"para reducir dependencia."
        )
    elif n_channels >= 3 and concentration < 0.4:
        insights.append(
            f"✅ **Portfolio balanceado**: Las ventas están bien distribuidas entre "
            f"canales ({n_channels} canales activos), reduciendo riesgo de concentración."
        )
    
    return insights
