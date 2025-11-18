"""
Visualization functions for MMM analysis.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List


def plot_beta_coefficients(
    media_cols: List[str],
    beta_means: np.ndarray,
    title: str = "Coeficientes medios (beta) por canal"
) -> go.Figure:
    """
    Create bar chart of beta coefficients by channel.
    
    Args:
        media_cols: List of media channel names
        beta_means: Mean beta values from posterior
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    df = pd.DataFrame({
        "Canal": media_cols,
        "Coeficiente_beta": beta_means
    })
    
    fig = px.bar(
        df,
        x="Canal",
        y="Coeficiente_beta",
        title=title,
        text_auto=".3f",
        labels={"Coeficiente_beta": "Coeficiente β (escala estandarizada)"}
    )
    
    fig.update_traces(textposition='outside')
    
    return fig


def plot_incremental_sales(
    media_cols: List[str],
    contributions: Dict[str, float],
    title: str = "Ventas incrementales atribuibles por canal"
) -> go.Figure:
    """
    Create bar chart of incremental sales by channel.
    
    Args:
        media_cols: List of media channel names
        contributions: Dictionary mapping channel to contribution
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    contrib_values = [contributions[ch] for ch in media_cols]
    
    fig = px.bar(
        x=media_cols,
        y=contrib_values,
        labels={"x": "Canal", "y": "Ventas incrementales"},
        title=title,
        text_auto=".2f"
    )
    
    fig.update_traces(textposition='outside')
    
    return fig


def plot_contribution_pie(
    media_cols: List[str],
    contributions: Dict[str, float],
    baseline_total: float,
    residual_total: float,
    title: str = "Contribución porcentual a las ventas"
) -> go.Figure:
    """
    Create pie chart showing contribution breakdown.
    
    Args:
        media_cols: List of media channel names
        contributions: Dictionary mapping channel to contribution
        baseline_total: Baseline contribution value
        residual_total: Residual value
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    components = media_cols + ["Baseline", "Residuo"]
    values = [contributions[ch] for ch in media_cols] + [baseline_total, residual_total]
    
    df = pd.DataFrame({
        "Componente": components,
        "Contribución": values
    })
    
    fig = px.pie(
        df,
        names="Componente",
        values="Contribución",
        title=title
    )
    
    return fig


def plot_waterfall(
    media_cols: List[str],
    contributions: Dict[str, float],
    baseline_total: float,
    residual_total: float,
    total_sales: float,
    title: str = "Cascada de contribución a las ventas"
) -> go.Figure:
    """
    Create waterfall chart showing sales decomposition.
    
    Args:
        media_cols: List of media channel names
        contributions: Dictionary mapping channel to contribution
        baseline_total: Baseline contribution value
        residual_total: Residual value
        total_sales: Total sales value
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    x_labels = ["Baseline"] + media_cols + ["Residuo", "Total ventas"]
    y_values = [baseline_total] + [contributions[ch] for ch in media_cols] + [residual_total, total_sales]
    measures = ["relative"] * (len(media_cols) + 2) + ["total"]
    
    fig = go.Figure(go.Waterfall(
        name="Contribución",
        orientation="v",
        x=x_labels,
        measure=measures,
        y=y_values,
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}}
    ))
    
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis_title="Componente",
        yaxis_title="Ventas"
    )
    
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_col: str = "Sales",
    title: str = "Ventas reales vs predichas"
) -> go.Figure:
    """
    Create line plot comparing actual vs predicted values.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        target_col: Name of target variable
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    idx = np.arange(len(y_true))
    
    df = pd.DataFrame({
        "Observación": idx,
        "Real": y_true,
        "Predicho": y_pred
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["Observación"],
        y=df["Real"],
        mode="lines+markers",
        name="Real",
        line=dict(color="blue")
    ))
    
    fig.add_trace(go.Scatter(
        x=df["Observación"],
        y=df["Predicho"],
        mode="lines+markers",
        name="Predicho",
        line=dict(color="red", dash="dash")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Observación",
        yaxis_title=target_col,
        hovermode="x unified"
    )
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuos del modelo"
) -> go.Figure:
    """
    Create residual plot.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    residuals = y_true - y_pred
    idx = np.arange(len(residuals))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=idx,
        y=residuals,
        mode="markers",
        name="Residuos",
        marker=dict(color="red", size=6)
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=title,
        xaxis_title="Observación",
        yaxis_title="Residuo",
        showlegend=False
    )
    
    return fig


def plot_contribution_table(
    contrib_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Format contribution DataFrame for display.
    
    Args:
        contrib_df: DataFrame with contribution metrics
        
    Returns:
        Styled DataFrame
    """
    format_dict = {
        "Coeficiente_gamma": "{:.6f}",
        "Contribución_total": "{:,.2f}",
        "Inversión_total": "{:,.2f}",
        "ROI": "{:.4f}",
        "ROAS": "{:.4f}",
        "Share_of_Sales": "{:.2%}"
    }
    
    # Only format columns that exist
    existing_format = {k: v for k, v in format_dict.items() if k in contrib_df.columns}
    
    return contrib_df.style.format(existing_format)


def plot_residuals_vs_predicted(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    title: str = "Residuos vs Predicción"
) -> go.Figure:
    """
    Create scatter plot of residuals vs predicted values for diagnostic.
    
    This plot helps detect:
    - Heteroscedasticity (non-constant variance)
    - Non-linearity
    - Outliers
    
    Args:
        y_pred: Predicted values
        residuals: Residuals (y_true - y_pred)
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode="markers",
        name="Residuos",
        marker=dict(
            color="steelblue",
            size=8,
            opacity=0.6,
            line=dict(width=1, color="white")
        )
    ))
    
    # Add zero reference line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        annotation_text="y=0",
        annotation_position="right"
    )
    
    # Add smoothing line (LOWESS-like)
    try:
        from scipy.interpolate import UnivariateSpline
        sorted_idx = np.argsort(y_pred)
        x_sorted = y_pred[sorted_idx]
        y_sorted = residuals[sorted_idx]
        
        # Only smooth if we have enough points
        if len(x_sorted) > 10:
            spline = UnivariateSpline(x_sorted, y_sorted, s=len(x_sorted), k=3)
            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 100)
            y_smooth = spline(x_smooth)
            
            fig.add_trace(go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode="lines",
                name="Tendencia",
                line=dict(color="orange", width=2)
            ))
    except:
        pass  # Skip smoothing if scipy not available or fails
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicción",
        yaxis_title="Residuo",
        hovermode="closest",
        showlegend=True,
        height=450
    )
    
    return fig


def plot_residuals_histogram(
    residuals: np.ndarray,
    title: str = "Distribución de Residuos"
) -> go.Figure:
    """
    Create histogram of residuals with normal distribution overlay.
    
    This plot helps check the normality assumption of residuals.
    
    Args:
        residuals: Residuals (y_true - y_pred)
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=residuals,
        name="Residuos",
        nbinsx=30,
        marker=dict(
            color="steelblue",
            line=dict(color="white", width=1)
        ),
        opacity=0.7,
        histnorm="probability density"
    ))
    
    # Overlay normal distribution
    mean = np.mean(residuals)
    std = np.std(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    normal_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal_pdf,
        mode="lines",
        name="Normal teórica",
        line=dict(color="red", width=2, dash="dash")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Residuo",
        yaxis_title="Densidad",
        showlegend=True,
        height=400,
        bargap=0.05
    )
    
    # Add stats annotation
    fig.add_annotation(
        text=f"Media: {mean:.4f}<br>Std: {std:.4f}",
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
        xanchor="right",
        yanchor="top"
    )
    
    return fig


def plot_qq_plot(
    residuals: np.ndarray,
    title: str = "Q-Q Plot de Residuos"
) -> go.Figure:
    """
    Create Q-Q (Quantile-Quantile) plot to assess normality of residuals.
    
    Points should lie roughly on the diagonal if residuals are normally distributed.
    
    Args:
        residuals: Residuals (y_true - y_pred)
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    from scipy import stats
    
    # Standardize residuals
    standardized = (residuals - np.mean(residuals)) / np.std(residuals)
    standardized = np.sort(standardized)
    
    # Theoretical quantiles
    n = len(standardized)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))
    
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=standardized,
        mode="markers",
        name="Datos",
        marker=dict(color="steelblue", size=6)
    ))
    
    # Add reference line (y=x)
    min_val = min(theoretical_quantiles.min(), standardized.min())
    max_val = max(theoretical_quantiles.max(), standardized.max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Normal teórica",
        line=dict(color="red", width=2, dash="dash")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Cuantiles Teóricos",
        yaxis_title="Cuantiles Observados",
        showlegend=True,
        height=450
    )
    
    return fig

