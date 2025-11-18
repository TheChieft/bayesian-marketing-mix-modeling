"""
Marketing Mix Modeling Dashboard
=================================
Streamlit UI for Bayesian MMM using PyMC.

This application provides an interactive interface for:
- Loading advertising data
- Configuring model parameters (adstock, saturation)
- Fitting Bayesian MMM models
- Analyzing contributions, ROI, and ROAS
- Visualizing results
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import mmm_core
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmm_core import data, transforms, model, metrics, viz

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(page_title="Marketing Mix Modeling (PyMC)", layout="wide")

# ============================================================================
# UI Header
# ============================================================================

st.title("🎯 Marketing Mix Modeling (Bayesiano con PyMC)")
st.markdown("""
Dashboard basado en PyMC que ajusta un modelo bayesiano de MMM usando:
- **Transformaciones de medios**: Adstock + saturación Hill
- **Estandarización**: Mejora convergencia del modelo
- **Inferencia variacional (ADVI)**: Rápida y eficiente
- **Análisis completo**: ROI, ROAS, contribuciones, y visualizaciones
""")

# ============================================================================
# Sidebar: Data Loading
# ============================================================================

st.sidebar.header("📊 Datos")

uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo CSV",
    type=["csv"],
    help="El archivo debe contener columnas de inversión en medios y ventas"
)

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Datos cargados correctamente")
    except Exception as e:
        st.sidebar.error(f"❌ Error al cargar archivo: {e}")
        st.stop()
else:
    st.sidebar.info("📝 No se cargó archivo. Usando datos sintéticos.")
    df_raw = data.generate_synthetic_data(n=200, seed=42)

# Sanitize column names
df, name_map = data.sanitize_columns(df_raw)

# Show renamed columns
changed = {k: v for k, v in name_map.items() if k != v}
if changed:
    with st.expander("ℹ️ Columnas renombradas (para uso seguro en el modelo)"):
        st.dataframe(
            pd.DataFrame(list(changed.items()), columns=["Original", "Renombrada"]),
            use_container_width=True
        )

# Data preview
st.subheader("📋 Vista previa de los datos")
st.dataframe(df.head(10), use_container_width=True)

# ============================================================================
# Sidebar: Model Configuration
# ============================================================================

st.sidebar.header("⚙️ Configuración del modelo")

# Target variable selection
default_target_idx = df.columns.get_loc("Sales_") if "Sales_" in df.columns else len(df.columns) - 1
target_col = st.sidebar.selectbox(
    "Variable objetivo (ventas)",
    df.columns,
    index=default_target_idx,
    help="Selecciona la columna que representa las ventas"
)

# Media variables selection
media_candidates = [c for c in df.columns if c != target_col]
default_media = [c for c in media_candidates if any(kw in c for kw in ["TV", "Radio", "Newspaper"])]

media_cols = st.sidebar.multiselect(
    "Variables de medios",
    media_candidates,
    default=default_media,
    help="Selecciona las columnas de inversión en medios"
)

# Transformation parameters
st.sidebar.subheader("🔧 Parámetros de transformación")

adstock_rate = st.sidebar.slider(
    "Tasa de Adstock (r)",
    min_value=0.0,
    max_value=0.9,
    value=0.1,
    step=0.05,
    help="Efecto de arrastre: ¿cuánto del impacto persiste al día siguiente?"
)

hill_gamma = st.sidebar.slider(
    "Curvatura Hill (γ)",
    min_value=0.5,
    max_value=3.0,
    value=1.5,
    step=0.1,
    help="Forma de la saturación: valores >1 crean forma S"
)

# Model fitting method
method = st.sidebar.selectbox(
    "Método de inferencia",
    options=["advi", "nuts"],
    index=0,
    help="ADVI es rápido (recomendado), NUTS es más preciso pero lento"
)

# ============================================================================
# Model Fitting
# ============================================================================

if st.sidebar.button("🚀 Ajustar modelo", type="primary"):
    if len(media_cols) == 0:
        st.error("❌ Selecciona al menos una variable de medios.")
        st.stop()
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Transformations
    status_text.text("⚙️ Aplicando transformaciones (Adstock + Hill)...")
    progress_bar.progress(20)
    
    df_trans, sat_cols = transforms.build_transformed_media(
        df, media_cols, adstock_rate, hill_gamma
    )
    
    with st.expander("🔍 Columnas transformadas"):
        st.write("**Columnas saturadas usadas en el modelo:**")
        st.write(sat_cols)
    
    # Prepare data
    X = df_trans[sat_cols].values.astype('float32')
    y = df_trans[target_col].values.astype('float32')
    
    # Step 2: Standardization
    status_text.text("📊 Estandarizando datos...")
    progress_bar.progress(30)
    
    X_scaled, y_scaled, scaler_X, scaler_y = transforms.standardize_data(X, y)
    
    # Step 3: Build and fit model
    status_text.text(f"🔮 Ajustando modelo bayesiano ({method.upper()})...")
    progress_bar.progress(40)
    
    mmm_model = model.build_mmm_model(X_scaled, y_scaled)
    idata = model.fit_mmm_model(
        mmm_model,
        method=method,
        draws=2000,
        tune=1000,
        n_advi=8000,
        random_seed=42
    )
    
    status_text.text("✅ Modelo ajustado correctamente")
    progress_bar.progress(60)
    
    # Step 4: Extract results
    status_text.text("📈 Calculando métricas y contribuciones...")
    progress_bar.progress(70)
    
    summary = model.get_posterior_summary(idata)
    beta_means = model.extract_beta_coefficients(idata)
    alpha_mean = summary.loc["alpha", "mean"]
    
    # Step 5: Predictions
    # Rebuild model for posterior predictive (PyMC requirement)
    mmm_model_pred = model.build_mmm_model(X_scaled, y_scaled)
    y_pred_scaled, _ = model.predict_posterior(mmm_model_pred, idata, X_scaled)
    
    # Transform back to original scale
    y_pred = transforms.inverse_transform_predictions(y_pred_scaled, scaler_y)
    y_true = y
    
    # Step 6: Metrics
    fit_metrics = metrics.compute_fit_metrics(y_true, y_pred)
    
    # Step 7: Contributions
    contrib_df, baseline_total, contributions = metrics.compute_contributions(
        X, beta_means, alpha_mean, scaler_X, scaler_y, media_cols
    )
    
    total_sales = float(y_true.sum())
    residual_total = metrics.compute_residual(y_true, baseline_total, contributions)
    
    # Step 8: ROI/ROAS
    contrib_df = metrics.compute_roi_roas(contrib_df, df, media_cols, total_sales)
    
    progress_bar.progress(100)
    status_text.text("✅ ¡Análisis completo!")
    
    # ========================================================================
    # Display Results
    # ========================================================================
    
    st.success("✅ Modelo ajustado exitosamente")
    
    # Metrics
    st.subheader("📊 Métricas de ajuste del modelo")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R²", f"{fit_metrics['R2']:.3f}" if not np.isnan(fit_metrics['R2']) else "N/A")
    with col2:
        st.metric("RMSE", f"{fit_metrics['RMSE']:,.3f}")
    with col3:
        st.metric("MAPE", f"{fit_metrics['MAPE']:.2f}%" if not np.isnan(fit_metrics['MAPE']) else "N/A")
    
    # Posterior summary
    with st.expander("🔬 Resumen bayesiano (ArviZ)"):
        st.dataframe(summary, use_container_width=True)
    
    # Beta coefficients
    st.subheader("📈 Coeficientes por canal")
    fig_betas = viz.plot_beta_coefficients(media_cols, beta_means)
    st.plotly_chart(fig_betas, use_container_width=True)
    
    # Contributions table
    st.subheader("💰 Contribuciones, ROI y ROAS por canal")
    
    # Format and display table
    st.dataframe(
        contrib_df.style.format({
            "Coeficiente_gamma": "{:.6f}",
            "Contribución_total": "{:,.2f}",
            "Inversión_total": "{:,.2f}",
            "ROI": "{:.4f}",
            "ROAS": "{:.4f}",
            "Share_of_Sales": "{:.2%}"
        }),
        use_container_width=True
    )
    
    # Visualizations
    st.subheader("📊 Visualizaciones de contribución")
    
    # Incremental sales
    fig_inc = viz.plot_incremental_sales(media_cols, contributions)
    st.plotly_chart(fig_inc, use_container_width=True)
    
    # Pie and waterfall side by side
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = viz.plot_contribution_pie(
            media_cols, contributions, baseline_total, residual_total
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_waterfall = viz.plot_waterfall(
            media_cols, contributions, baseline_total, residual_total, total_sales
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
    
    # Actual vs Predicted
    st.subheader("🎯 Ajuste del modelo: Real vs Predicho")
    fig_pred = viz.plot_actual_vs_predicted(y_true, y_pred, target_col)
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Store results in session state for potential download
    st.session_state['last_results'] = {
        'contrib_df': contrib_df,
        'fit_metrics': fit_metrics,
        'summary': summary
    }

else:
    st.info("👈 Configura el modelo en el panel lateral y haz clic en 'Ajustar modelo'")

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.caption("🎓 Proyecto académico - Marketing Mix Modeling Bayesiano con PyMC | Desarrollado por TheChieft")
