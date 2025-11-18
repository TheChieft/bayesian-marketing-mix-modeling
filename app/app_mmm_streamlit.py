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

# Initialize session state for persistence
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'media_cols' not in st.session_state:
    st.session_state.media_cols = []
if 'unit_scale' not in st.session_state:
    st.session_state.unit_scale = "original"

# Dataset mode selector
dataset_mode = st.sidebar.radio(
    "Modo de dataset",
    ["📁 Usar dataset de ejemplo (Basemediosfinal.csv)", "📤 Subir dataset propio (CSV)"],
    help="Elige si quieres usar el dataset de demostración o subir tu propio archivo"
)

df_raw = None
data_source = ""

if dataset_mode.startswith("📁"):
    # Use example dataset
    try:
        df_raw = data.load_example_dataset()
        data_source = "Dataset de ejemplo (Basemediosfinal.csv)"
        st.sidebar.success("✅ Dataset de ejemplo cargado")
    except FileNotFoundError as e:
        st.sidebar.error(f"❌ {e}")
        st.error("El dataset de ejemplo no está disponible. Por favor, súbelo o usa datos sintéticos.")
        df_raw = data.generate_synthetic_data(n=200, seed=42)
        data_source = "Datos sintéticos generados"
else:
    # Upload own dataset
    uploaded_file = st.sidebar.file_uploader(
        "Sube tu archivo CSV",
        type=["csv"],
        help="El archivo debe contener columnas de inversión en medios y ventas"
    )
    
    if uploaded_file:
        try:
            df_raw = pd.read_csv(uploaded_file)
            data_source = f"Archivo subido: {uploaded_file.name}"
            
            # Validate schema
            is_valid, error_msg, numeric_cols = data.validate_dataset_schema(df_raw)
            
            if not is_valid:
                st.sidebar.error("❌ Validación fallida")
                st.error(f"### ⚠️ Problema con el dataset\n\n{error_msg}")
                st.info("""
                **💡 Consejo**: Tu archivo CSV debe tener:
                - Al menos 10 filas de datos
                - Al menos 2 columnas numéricas (1 para ventas + 1+ para medios)
                - Sin columnas con >50% de valores faltantes
                
                **Ejemplo de formato correcto:**
                ```
                Fecha,TV_Budget,Radio_Budget,Digital_Budget,Sales
                2024-01-01,1000,500,300,5000
                2024-01-02,1200,600,350,5500
                ...
                ```
                """)
                st.stop()
            else:
                st.sidebar.success(f"✅ Datos válidos ({len(df_raw)} filas, {len(numeric_cols)} cols numéricas)")
        except Exception as e:
            st.sidebar.error(f"❌ Error al cargar: {e}")
            st.stop()
    else:
        st.sidebar.info("📝 Esperando archivo...")
        st.info("👈 Por favor, sube un archivo CSV para continuar.")
        st.stop()

# Sanitize column names
df, name_map = data.sanitize_columns(df_raw)

# Show renamed columns
changed = {k: v for k, v in name_map.items() if k != v}
if changed:
    with st.expander("ℹ️ Columnas renombradas (para uso seguro en el modelo)"):
        rename_df = pd.DataFrame(list(changed.items()), columns=["Nombre original", "Nombre en el modelo"])
        st.dataframe(rename_df, use_container_width=True)
        st.caption("Las columnas se renombran automáticamente para evitar caracteres especiales que podrían causar errores.")

# Data preview with info
st.subheader("📋 Vista previa de los datos")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Filas", len(df))
with col2:
    st.metric("Columnas", len(df.columns))
with col3:
    st.metric("Fuente", data_source, delta=None)

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
st.sidebar.subheader("🔬 Método de inferencia")

method = st.sidebar.selectbox(
    "Algoritmo",
    options=["advi", "nuts"],
    index=0,
    format_func=lambda x: {
        "advi": "ADVI (Rápido, recomendado)",
        "nuts": "NUTS (Preciso, más lento)"
    }[x],
    help="ADVI: Inferencia variacional (minutos). NUTS: MCMC (puede tomar 10-30 min)"
)

if method == "nuts":
    st.sidebar.warning(
        "⚠️ **NUTS es experimental**: Más preciso pero mucho más lento. "
        "Se recomienda solo para datasets pequeños (<100 filas)."
    )
    draws_nuts = st.sidebar.slider(
        "Draws NUTS",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100,
        help="Número de muestras posteriores (menos muestras = más rápido)"
    )
else:
    draws_nuts = 2000  # Default for ADVI

# Train/Test Split
st.sidebar.subheader("🎯 Train/Test Split")

enable_validation = st.sidebar.checkbox(
    "Habilitar validación",
    value=False,
    help="Divide datos en entrenamiento y prueba para evaluar generalización"
)

if enable_validation:
    train_pct = st.sidebar.slider(
        "% Datos de entrenamiento",
        min_value=60,
        max_value=90,
        value=70,
        step=5,
        help="Porcentaje de datos para entrenamiento (resto para validación)"
    )
    test_pct = 100 - train_pct
    st.sidebar.info(f"📊 Train: {train_pct}% | Test: {test_pct}%")
else:
    train_pct = 100

# Unit scale selector
st.sidebar.subheader("📊 Escala de unidades")

unit_scale = st.sidebar.selectbox(
    "Mostrar valores en",
    options=["original", "thousands", "millions"],
    format_func=lambda x: {
        "original": "Unidades originales",
        "thousands": "Miles",
        "millions": "Millones"
    }[x],
    index=0,
    help="Escala para mostrar contribuciones e inversiones"
)

currency = st.sidebar.text_input(
    "Moneda",
    value="COP",
    help="Código de moneda (ej: COP, USD, EUR)"
)

st.session_state.unit_scale = unit_scale

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
    
    # Step 2.5: Train/Test Split (if enabled)
    if enable_validation:
        status_text.text(f"✂️ Dividiendo datos ({train_pct}% train, {test_pct}% test)...")
        test_size = (100 - train_pct) / 100
        X_train, X_test, y_train, y_test = model.split_train_test(
            X_scaled, y_scaled, test_size=test_size, shuffle=False
        )
    else:
        X_train, y_train = X_scaled, y_scaled
        X_test, y_test = None, None
    
    # Step 3: Build and fit model
    status_text.text(f"🔮 Ajustando modelo bayesiano ({method.upper()})...")
    if method == "nuts":
        st.warning("⏱️ NUTS puede tomar varios minutos. Por favor espera...")
    progress_bar.progress(40)
    
    if enable_validation:
        # Use validation-aware fitting
        mmm_model, idata, validation_metrics = model.fit_mmm_with_validation(
            X_train, y_train, X_test, y_test,
            method=method,
            draws=draws_nuts if method == "nuts" else 2000,
            tune=1000 if method == "nuts" else 1000,
            n_advi=8000,
            random_seed=42
        )
    else:
        # Standard fitting
        mmm_model = model.build_mmm_model(X_train, y_train)
        idata = model.fit_mmm_model(
            mmm_model,
            method=method,
            draws=draws_nuts if method == "nuts" else 2000,
            tune=1000 if method == "nuts" else 1000,
            n_advi=8000,
            random_seed=42
        )
        validation_metrics = {}
    
    status_text.text("✅ Modelo ajustado correctamente")
    progress_bar.progress(60)
    
    # Step 4: Extract results
    status_text.text("📈 Calculando métricas y contribuciones...")
    progress_bar.progress(70)
    
    summary = model.get_posterior_summary(idata)
    beta_means = model.extract_beta_coefficients(idata)
    alpha_mean = summary.loc["alpha", "mean"]
    
    # Step 5: Predictions (on full dataset for visualization)
    # Rebuild model for posterior predictive (PyMC requirement)
    mmm_model_pred = model.build_mmm_model(X_scaled, y_scaled)
    y_pred_scaled, _ = model.predict_posterior(mmm_model_pred, idata, X_scaled)
    
    # Transform back to original scale
    y_pred = transforms.inverse_transform_predictions(y_pred_scaled, scaler_y)
    y_true = y
    residuals = y_true - y_pred
    
    # Step 6: Metrics (full dataset)
    fit_metrics = metrics.compute_fit_metrics(y_true, y_pred)
    
    # Step 7: Contributions
    contrib_df, baseline_total, contributions = metrics.compute_contributions(
        X, beta_means, alpha_mean, scaler_X, scaler_y, media_cols
    )
    
    total_sales = float(y_true.sum())
    residual_total = metrics.compute_residual(y_true, baseline_total, contributions)
    
    # Step 8: ROI/ROAS
    contrib_df = metrics.compute_roi_roas(contrib_df, df, media_cols, total_sales)
    
    # Step 9: Contribution uncertainty (credibility intervals)
    status_text.text("🔍 Calculando intervalos de credibilidad (90%)...")
    progress_bar.progress(90)
    
    uncertainty_df = metrics.compute_contribution_uncertainty(
        X, idata, scaler_X, scaler_y, media_cols, ci_level=0.90
    )
    
    # Merge uncertainty into contrib_df
    contrib_df = contrib_df.merge(uncertainty_df[['Canal', 'CI_lower', 'CI_upper', 'CI_width']], on='Canal', how='left')
    
    progress_bar.progress(100)
    status_text.text("✅ ¡Análisis completo!")
    
    # ========================================================================
    # Display Results
    # ========================================================================
    
    st.success("✅ Modelo ajustado exitosamente")
    
    # Metrics
    st.subheader("📊 Métricas de ajuste del modelo")
    
    if enable_validation and validation_metrics:
        # Show both train and test metrics
        st.markdown("**Métricas In-Sample (Training):**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Train", f"{validation_metrics['train_r2']:.3f}")
        with col2:
            st.metric("RMSE Train", f"{validation_metrics['train_rmse']:,.3f}")
        with col3:
            st.metric("MAPE Train", f"{validation_metrics['train_mape']:.2f}%")
        
        st.markdown("**Métricas Out-of-Sample (Test):**")
        col4, col5, col6 = st.columns(3)
        with col4:
            r2_delta = validation_metrics['test_r2'] - validation_metrics['train_r2']
            st.metric(
                "R² Test",
                f"{validation_metrics['test_r2']:.3f}",
                delta=f"{r2_delta:.3f}",
                delta_color="normal"
            )
        with col5:
            rmse_delta = validation_metrics['test_rmse'] - validation_metrics['train_rmse']
            st.metric(
                "RMSE Test",
                f"{validation_metrics['test_rmse']:,.3f}",
                delta=f"{rmse_delta:,.3f}",
                delta_color="inverse"
            )
        with col6:
            st.metric("MAPE Test", f"{validation_metrics['test_mape']:.2f}%")
        
        # Overfitting analysis
        if abs(r2_delta) > 0.15:
            st.warning(
                f"⚠️ **Posible overfitting**: R² cae {abs(r2_delta):.3f} puntos entre train y test. "
                "Considera simplificar el modelo o aumentar regularización."
            )
        elif abs(r2_delta) < 0.05:
            st.success("✅ **Buen ajuste**: Las métricas train/test son consistentes. No hay señales de overfitting.")
    else:
        # Show full dataset metrics
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
    
    # Apply unit scaling
    unit_label = metrics.get_unit_label(unit_scale, currency)
    
    contrib_df_display = contrib_df.copy()
    contrib_df_display["Contribución_total"] = contrib_df_display["Contribución_total"].apply(
        lambda x: metrics.scale_to_units(x, unit_scale)
    )
    contrib_df_display["Inversión_total"] = contrib_df_display["Inversión_total"].apply(
        lambda x: metrics.scale_to_units(x, unit_scale)
    )
    
    # Scale CI if present
    if 'CI_lower' in contrib_df_display.columns:
        contrib_df_display["CI_lower"] = contrib_df_display["CI_lower"].apply(
            lambda x: metrics.scale_to_units(x, unit_scale)
        )
        contrib_df_display["CI_upper"] = contrib_df_display["CI_upper"].apply(
            lambda x: metrics.scale_to_units(x, unit_scale)
        )
        contrib_df_display["CI_width"] = contrib_df_display["CI_width"].apply(
            lambda x: metrics.scale_to_units(x, unit_scale)
        )
    
    # Format and display table
    format_dict = {
        "Coeficiente_gamma": "{:.6f}",
        "Contribución_total": "{:,.2f}",
        "Inversión_total": "{:,.2f}",
        "ROI": "{:.4f}",
        "ROAS": "{:.4f}",
        "Share_of_Sales": "{:.2%}"
    }
    
    if 'CI_lower' in contrib_df_display.columns:
        format_dict.update({
            "CI_lower": "{:,.2f}",
            "CI_upper": "{:,.2f}",
            "CI_width": "{:,.2f}"
        })
    
    st.dataframe(
        contrib_df_display.style.format(format_dict),
        use_container_width=True
    )
    
    if 'CI_lower' in contrib_df_display.columns:
        st.info(
            "💡 **Intervalos de Credibilidad (IC 90%)**: CI_lower y CI_upper muestran el rango "
            "donde se encuentra la contribución real con 90% de probabilidad bayesiana. "
            "Rangos amplios indican mayor incertidumbre."
        )
    
    st.caption(f"💡 Valores mostrados en: **{unit_label}**")
    
    # ========================================================================
    # Business Insights Section
    # ========================================================================
    
    st.subheader("📌 Insights de Negocio")
    st.markdown("Análisis automático basado en los resultados del modelo:")
    
    # Generate insights
    total_budget = contrib_df["Inversión_total"].sum()
    insights = metrics.generate_business_insights(contrib_df, total_sales, total_budget, uncertainty_df)
    
    # Display insights
    for insight in insights:
        st.markdown(insight)
    
    # Download report button
    st.markdown("---")
    
    # Generate report content
    report_content = f"""# Reporte de Marketing Mix Modeling
## Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Configuración del Modelo

- **Dataset**: {data_source}
- **Variable objetivo**: {target_col}
- **Variables de medios**: {', '.join(media_cols)}
- **Tasa de Adstock**: {adstock_rate}
- **Curvatura Hill (γ)**: {hill_gamma}
- **Método de inferencia**: {method.upper()}
- **Escala de unidades**: {unit_label}

---

## 📈 Métricas de Ajuste

- **R²**: {fit_metrics['R2']:.3f}
- **RMSE**: {fit_metrics['RMSE']:,.3f}
- **MAPE**: {fit_metrics['MAPE']:.2f}%

---

## 💰 Contribuciones por Canal

{contrib_df.to_markdown(index=False)}

---

## 📌 Insights de Negocio

"""
    
    for i, insight in enumerate(insights, 1):
        # Remove markdown formatting for plain text
        clean_insight = insight.replace("**", "").replace("*", "")
        report_content += f"{i}. {clean_insight}\n"
    
    report_content += f"""

---

## 📊 Resumen Ejecutivo

### Ventas Totales
- Total: {metrics.scale_to_units(total_sales, unit_scale):,.2f} {unit_label}
- Baseline: {metrics.scale_to_units(baseline_total, unit_scale):,.2f} {unit_label} ({baseline_total/total_sales*100:.1f}%)

### Performance por Canal
"""
    
    for _, row in contrib_df.iterrows():
        contrib_scaled = metrics.scale_to_units(row['Contribución_total'], unit_scale)
        invest_scaled = metrics.scale_to_units(row['Inversión_total'], unit_scale)
        report_content += f"""
**{row['Canal']}**
- Contribución: {contrib_scaled:,.2f} {unit_label} ({row['Share_of_Sales']*100:.1f}% de ventas)
- Inversión: {invest_scaled:,.2f} {unit_label}
- ROI: {row['ROI']*100:.1f}%
- ROAS: {row['ROAS']:.2f}
"""
    
    report_content += f"""

---

*Reporte generado automáticamente por Marketing Mix Modeling Dashboard*
*Desarrollado por TheChieft*
"""
    
    # Download button
    st.download_button(
        label="📥 Descargar Reporte Completo (Markdown)",
        data=report_content,
        file_name=f"mmm_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        help="Descarga un reporte completo con métricas, contribuciones e insights"
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
    
    # Diagnostic plots
    with st.expander("🔬 Diagnósticos del Modelo", expanded=False):
        st.markdown("""
        **¿Qué evalúan estos gráficos?**
        - **Residuos vs Predicción**: Detecta heterocedasticidad (varianza no constante) y no-linealidad.
          Idealmente, los puntos deben estar dispersos aleatoriamente alrededor de 0.
        - **Histograma de Residuos**: Verifica normalidad. Debe parecerse a una campana de Gauss.
        - **Q-Q Plot**: Confirma normalidad. Los puntos deben seguir la línea roja.
        """)
        
        col_diag1, col_diag2 = st.columns(2)
        
        with col_diag1:
            fig_resid_vs_pred = viz.plot_residuals_vs_predicted(y_pred, residuals)
            st.plotly_chart(fig_resid_vs_pred, use_container_width=True)
        
        with col_diag2:
            fig_resid_hist = viz.plot_residuals_histogram(residuals)
            st.plotly_chart(fig_resid_hist, use_container_width=True)
        
        # Q-Q Plot
        try:
            fig_qq = viz.plot_qq_plot(residuals)
            st.plotly_chart(fig_qq, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ No se pudo generar Q-Q plot: {e}")
    
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
