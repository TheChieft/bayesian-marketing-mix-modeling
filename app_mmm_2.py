# app_mmm_pymc.py
# -----------------------------------
# Dashboard de Marketing Mix Modeling (Bayesiano con PyMC)
# -----------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import re

st.set_page_config(page_title="Marketing Mix Modeling (PyMC)", layout="wide")

# =========================
# Utilidades
# =========================

def sanitize_columns(df: pd.DataFrame):
    """
    Convierte nombres de columnas a identificadores seguros:
    - Sustituye todo lo que no sea [a-zA-Z0-9_] por "_"
    - Si el nombre empieza por dígito, le antepone "X_"
    Devuelve: df_renombrado, mapping_original_a_seguro
    """
    mapping = {}
    for c in df.columns:
        safe = re.sub(r"\W+", "_", str(c))
        if safe and safe[0].isdigit():
            safe = f"X_{safe}"
        mapping[c] = safe
    return df.rename(columns=mapping), mapping


def adstock(s: pd.Series, r: float = 0.1) -> pd.Series:
    """Adstock simple con tasa de decaimiento r."""
    o = np.zeros(len(s))
    o[0] = s.iloc[0]
    for t in range(1, len(s)):
        o[t] = s.iloc[t] + r * o[t - 1]
    return pd.Series(o, index=s.index)


def hill(x, a: float = 1.0, t: float = 1.0, g: float = 1.5):
    """
    Función de saturación tipo Hill:
        f(x) = a * x^g / (t^g + x^g)
    """
    x = np.array(x, dtype=float)
    return a * (x**g) / (t**g + x**g)


def build_transforms(df: pd.DataFrame, media_cols, adstock_rate: float, g_hill: float):
    """
    Aplica Adstock + Hill a cada columna de medios.
    Devuelve df_trans, lista de columnas saturadas.
    """
    df_trans = df.copy()
    sat_cols = []
    for c in media_cols:
        ad_col = f"{c}_ad"
        sat_col = f"{c}_sat"
        df_trans[ad_col] = adstock(df[c], r=adstock_rate)
        theta_c = df_trans[ad_col].mean() if df_trans[ad_col].mean() > 0 else 1.0
        df_trans[sat_col] = hill(df_trans[ad_col], a=1.0, t=theta_c, g=g_hill)
        sat_cols.append(sat_col)
    return df_trans, sat_cols


# =========================
# UI
# =========================

st.title("Marketing Mix Modeling (Bayesiano con PyMC)")
st.markdown("""
Tablero basado en PyMC que ajusta un modelo bayesiano de MMM usando:
- Transformaciones de medios (Adstock + saturación Hill).
- Estandarización de variables.
- Inferencia variacional (ADVI) para rapidez.
Ahora también incluye ROI, ROAS, contribución a las ventas por canal, gráfico de torta y cascada.
""")

# -------------------------------------------------------
# 1. Carga de datos
# -------------------------------------------------------
st.sidebar.header("Datos")

uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV (Advertising)", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.success("Datos cargados correctamente.")
else:
    st.info("No se cargó ningún archivo. Generando datos sintéticos estilo Advertising.")
    np.random.seed(42)
    n = 200
    tv = np.random.uniform(0, 300, n)
    radio = np.random.uniform(0, 50, n)
    news = np.random.uniform(0, 50, n)
    baseline_true = 5.0
    sales = baseline_true + 0.04 * tv + 0.3 * radio + 0.02 * news + np.random.normal(0, 1.0, n)
    df_raw = pd.DataFrame({
        "Dia": np.arange(1, n + 1),
        "TV Ad Budget ($)": tv,
        "Radio Ad Budget ($)": radio,
        "Newspaper Ad Budget ($)": news,
        "Sales ($)": sales,
    })

# Sanitizar nombres de columnas
df, name_map = sanitize_columns(df_raw)
changed = {k: v for k, v in name_map.items() if k != v}
if changed:
    st.info("Se renombraron columnas para usar en el modelo (sin espacios ni símbolos):")
    st.write(pd.DataFrame(list(changed.items()), columns=["Nombre original", "Nombre en el modelo"]))

st.subheader("Vista previa de los datos")
st.dataframe(df.head())

# -------------------------------------------------------
# 2. Selección de columnas y parámetros de transformación
# -------------------------------------------------------
st.sidebar.header("Configuración del modelo")

default_target_idx = df.columns.get_loc("Sales_") if "Sales_" in df.columns else len(df.columns) - 1
target_col = st.sidebar.selectbox("Variable objetivo (ventas)", df.columns, index=default_target_idx)

media_candidates = [c for c in df.columns if c != target_col]
default_media = [c for c in media_candidates if "TV" in c or "Radio" in c or "Newspaper" in c]
media_cols = st.sidebar.multiselect("Variables de medios", media_candidates, default=default_media)

adstock_rate = st.sidebar.slider("Tasa de Adstock (r)", 0.0, 0.9, 0.1, 0.05)
g_hill = st.sidebar.slider("Curvatura Hill (g)", 0.5, 3.0, 1.5, 0.1)

# -------------------------------------------------------
# 3. Ajuste con PyMC
# -------------------------------------------------------
if st.sidebar.button("Ajustar modelo"):
    if len(media_cols) == 0:
        st.error("Selecciona al menos una variable de medios.")
    else:
        # 3.1 Transformaciones
        df_trans, sat_cols = build_transforms(df, media_cols, adstock_rate, g_hill)

        st.subheader("Transformaciones de medios")
        st.write("Columnas saturadas usadas en el modelo:")
        st.write(sat_cols)

        X = df_trans[sat_cols].values.astype("float32")
        y = df_trans[target_col].values.astype("float32")

        # 3.2 Estandarización
        scalerX = StandardScaler()
        X_s = scalerX.fit_transform(X).astype("float32")

        scalerY = StandardScaler()
        y_s = scalerY.fit_transform(y.reshape(-1, 1)).ravel().astype("float32")

        st.write(f"Shape X_s: {X_s.shape}, Shape y_s: {y_s.shape}")

        # 3.3 Modelo PyMC
        with st.spinner("Ejecutando inferencia bayesiana (ADVI en PyMC)..."):
            with pm.Model() as mmm:
                alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)
                betas = pm.TruncatedNormal(
                    "betas", mu=0.0, sigma=5.0, lower=0.0, shape=X_s.shape[1]
                )
                sigma = pm.HalfNormal("sigma", sigma=2.0)

                mu = alpha + pm.math.dot(X_s, betas)
                y_obs = pm.StudentT("y_obs", nu=5, mu=mu, sigma=sigma, observed=y_s)

                approx = pm.fit(
                    n=8000,
                    method="advi",
                    random_seed=42,
                    progressbar=False,
                )

                idata = approx.sample(2000)

        st.success("Modelo ajustado correctamente.")

        # -------------------------------------------------------
        # 4. Resumen bayesiano
        # -------------------------------------------------------
        st.subheader("Resumen bayesiano (ArviZ)")
        summary = az.summary(idata, var_names=["alpha", "betas", "sigma"])
        st.dataframe(summary)

        # Betas medias
        beta_summary = az.summary(idata, var_names=["betas"])
        beta_means = beta_summary["mean"].values  # vector (K,)
        betas_df = pd.DataFrame({
            "Canal": media_cols,
            "Coeficiente_beta_std": beta_means
        })

        fig_betas = px.bar(
            betas_df, x="Canal", y="Coeficiente_beta_std",
            title="Coeficientes medios (beta) por canal (escala estandarizada)",
            text_auto=".3f"
        )
        st.plotly_chart(fig_betas, use_container_width=True)

        # -------------------------------------------------------
        # 5. Posterior predictivo y métricas
        # -------------------------------------------------------
        with pm.Model() as mmm:
            alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)
            betas = pm.TruncatedNormal(
                "betas", mu=0.0, sigma=5.0, lower=0.0, shape=X_s.shape[1]
            )
            sigma = pm.HalfNormal("sigma", sigma=2.0)
            mu = alpha + pm.math.dot(X_s, betas)
            y_obs = pm.StudentT("y_obs", nu=5, mu=mu, sigma=sigma, observed=y_s)

            ppc = pm.sample_posterior_predictive(
                idata, var_names=["y_obs"], random_seed=42, progressbar=False
            )

        ppc_arr = ppc.posterior_predictive["y_obs"].values   # (chains, draws, N)
        ppc_mat = ppc_arr.reshape(-1, ppc_arr.shape[-1])     # (samples, N)
        ppc_mean_std = ppc_mat.mean(axis=0)

        # De vuelta a escala original
        y_pred = ppc_mean_std * scalerY.scale_[0] + scalerY.mean_[0]
        y_true = y

        residuals = y_true - y_pred
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - y_true.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        mask = y_true != 0
        mape = np.mean(np.abs(residuals[mask] / y_true[mask])) * 100 if mask.sum() > 0 else np.nan

        st.subheader("Métricas de ajuste del modelo")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R²", f"{r2:.3f}" if not np.isnan(r2) else "NA")
        with col2:
            st.metric("RMSE", f"{rmse:,.3f}")
        with col3:
            st.metric("MAPE", f"{mape:.2f} %" if not np.isnan(mape) else "NA")

        # -------------------------------------------------------
        # 6. Contribuciones, ROI y ROAS
        # -------------------------------------------------------
        st.subheader("Contribuciones por canal, ROI y ROAS")

        alpha_mean = summary.loc["alpha", "mean"]
        sigma_y = scalerY.scale_[0]
        mu_y = scalerY.mean_[0]
        sigma_x = scalerX.scale_
        mu_x = scalerX.mean_

        # Coeficientes en escala original de X (saturada)
        gamma = sigma_y * beta_means / sigma_x  # vector (K,)

        # Constante en escala original
        const = mu_y + sigma_y * alpha_mean - np.sum(gamma * mu_x)
        baseline_total = const * len(df_trans)

        contrib_dict = {}
        for j, f in enumerate(media_cols):
            Xj = X[:, j]  # columna saturada correspondiente
            contrib_dict[f] = float(gamma[j] * Xj.sum())

        total_sales = float(y_true.sum())
        residual_total = total_sales - (baseline_total + sum(contrib_dict.values()))

        # Ajustar residuo muy pequeño
        if abs(residual_total) < 1e-6 * max(total_sales, 1.0):
            residual_total = 0.0

        # ROI simple y ROAS
        roi_list = []
        roas_list = []
        inv_list = []
        for f in media_cols:
            spend = float(df[f].sum())
            inv_list.append(spend)
            contrib_j = contrib_dict[f]
            roi_val = contrib_j / spend if spend != 0 else np.nan
            roas_val = contrib_j / total_sales if total_sales != 0 else np.nan
            roi_list.append(roi_val)
            roas_list.append(roas_val)

        contrib_df = pd.DataFrame({
            "Canal": media_cols,
            "Coeficiente_gamma": gamma,
            "Contribución_total": [contrib_dict[f] for f in media_cols],
            "Inversión_total": inv_list,
            "ROI_simple": roi_list,
            "ROAS": roas_list,
        })

        st.dataframe(contrib_df.style.format({
            "Coeficiente_gamma": "{:.6f}",
            "Contribución_total": "{:,.2f}",
            "Inversión_total": "{:,.2f}",
            "ROI_simple": "{:.4f}",
            "ROAS": "{:.4f}",
        }))

        # -------------------------------------------------------
        # 7. Gráficos: incremental, torta, cascada
        # -------------------------------------------------------
        st.subheader("Gráficos de contribución e incremental")

        # Ventas incrementales por canal (barra)
        fig_inc = px.bar(
            x=media_cols,
            y=[contrib_dict[f] for f in media_cols],
            labels={"x": "Canal", "y": "Ventas incrementales atribuibles"},
            title="Ventas incrementales atribuibles por canal",
            text_auto=".2f"
        )
        st.plotly_chart(fig_inc, use_container_width=True)

        colg1, colg2 = st.columns(2)

        with colg1:
            pie_df = pd.DataFrame({
                "Componente": media_cols + ["Baseline", "Residuo"],
                "Contribución": [contrib_dict[f] for f in media_cols] + [baseline_total, residual_total],
            })
            fig_pie = px.pie(
                pie_df,
                names="Componente",
                values="Contribución",
                title="Contribución porcentual a las ventas (incluye Baseline y Residuo)"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with colg2:
            wf_x = ["Baseline"] + media_cols + ["Residuo", "Total ventas"]
            wf_y = [baseline_total] + [contrib_dict[f] for f in media_cols] + [residual_total, total_sales]
            wf_measure = ["relative"] * (len(media_cols) + 1) + ["relative", "total"]

            fig_wf = go.Figure(go.Waterfall(
                name="Contribución",
                orientation="v",
                x=wf_x,
                measure=wf_measure,
                y=wf_y,
                textposition="outside",
            ))
            fig_wf.update_layout(
                title="Cascada de contribución a las ventas",
                showlegend=False
            )
            st.plotly_chart(fig_wf, use_container_width=True)

        # -------------------------------------------------------
        # 8. Gráfico Real vs Predicho
        # -------------------------------------------------------
        st.subheader("Real vs Predicho")

        idx = np.arange(len(y_true))
        df_pred = pd.DataFrame({
            "Observación": idx,
            "Real": y_true,
            "Predicho": y_pred
        })

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=df_pred["Observación"],
            y=df_pred["Real"],
            mode="lines+markers",
            name="Real"
        ))
        fig_pred.add_trace(go.Scatter(
            x=df_pred["Observación"],
            y=df_pred["Predicho"],
            mode="lines+markers",
            name="Predicho"
        ))
        fig_pred.update_layout(
            title="Ventas reales vs predichas",
            xaxis_title="Observación",
            yaxis_title=target_col
        )
        st.plotly_chart(fig_pred, use_container_width=True)

else:
    st.warning("Selecciona variables y haz clic en 'Ajustar modelo' para ver resultados.")

# -------------------------------------------------------
# Fin
# -------------------------------------------------------
st.markdown("---")
st.caption("Proyecto académico - DeFi | MMM Bayesiano con PyMC (modelo + contribuciones)")
