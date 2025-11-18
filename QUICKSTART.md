# 🚀 Quick Start Guide

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/TheChieft/bayesian-marketing-mix-modeling.git
cd bayesian-marketing-mix-modeling

# 2. Crear y activar entorno conda
conda create -n mmm_bayes python=3.10
conda activate mmm_bayes

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Uso Rápido

### Opción 1: Ejecutar Dashboard Streamlit

```bash
# Forma rápida
./run_app.sh

# O manualmente
conda activate mmm_bayes
streamlit run app/app_mmm_streamlit.py
```

**Nuevas características (Fase 3, 4 & 5):**
- 📁 **Selector de dataset**: Elige entre ejemplo incluido o subir tu propio CSV
- ✅ **Validación automática**: Verifica que tu CSV tenga el formato correcto
- 📊 **Escala de unidades**: Muestra valores en unidades originales, miles o millones
- 💡 **Insights automáticos**: Análisis de negocio generado automáticamente
- 📥 **Reporte descargable**: Descarga todo el análisis en formato Markdown
- 🎯 **Train/Test split**: Valida el modelo con datos de prueba (60-90% configurable)
- 🔬 **Diagnósticos estadísticos**: Residuos, Q-Q plot, heteroscedasticidad
- 📉 **Intervalos de credibilidad**: IC 90% para cada canal (cuantifica incertidumbre)
- 🔮 **Opción NUTS**: Inferencia MCMC más precisa (experimental, para datasets pequeños)

### Opción 2: Usar mmm_core en Python

```python
from mmm_core import data, transforms, model, metrics, viz

# Cargar datos
df = data.load_base_data("data/Basemediosfinal.csv")
df, mapping = data.sanitize_columns(df)

# Transformar
df_trans, sat_cols = transforms.build_transformed_media(
    df, media_cols, adstock_rate=0.1, hill_gamma=1.5
)

# Preparar y escalar
X = df_trans[sat_cols].values
y = df_trans[target_col].values
X_scaled, y_scaled, scaler_X, scaler_y = transforms.standardize_data(X, y)

# Modelar
mmm = model.build_mmm_model(X_scaled, y_scaled)
idata = model.fit_mmm_model(mmm, method='advi')

# Predecir
y_pred_scaled, _ = model.predict_posterior(mmm, idata)
y_pred = transforms.inverse_transform_predictions(y_pred_scaled, scaler_y)

# Métricas
fit_metrics = metrics.compute_fit_metrics(y, y_pred)
contrib_df, baseline, contributions = metrics.compute_contributions(
    X, beta_means, alpha_mean, scaler_X, scaler_y, media_cols
)
contrib_df = metrics.compute_roi_roas(contrib_df, df, media_cols, total_sales)

# 🚀 Quick Start

Guía mínima para instalar y ejecutar la app. Para información técnica y ejemplos extensos, consulta `README.md`.

## Instalación rápida

```bash
# Clonar
git clone https://github.com/TheChieft/bayesian-marketing-mix-modeling.git
cd bayesian-marketing-mix-modeling

# Crear entorno (conda o venv)
conda create -n mmm_bayes python=3.10 -y
conda activate mmm_bayes

# Instalar dependencias
pip install -r requirements.txt
```

## Ejecutar la aplicación Streamlit

```bash
# Rápido (script que limpia caches y arranca)
./run_app.sh

# O manualmente
conda activate mmm_bayes
streamlit run app/app_mmm_streamlit.py
```

## Usar la librería (ejemplo mínimo)

```python
from mmm_core import data, transforms, model

df = data.load_base_data('data/Basemediosfinal.csv')
df_trans, sat_cols = transforms.build_transformed_media(df, media_cols=['TV','Radio'], adstock_rate=0.1, hill_gamma=1.5)
X = df_trans[sat_cols].values
y = df_trans['Sales'].values
X_scaled, y_scaled, scaler_X, scaler_y = transforms.standardize_data(X, y)

mmm = model.build_mmm_model(X_scaled, y_scaled)
idata = model.fit_mmm_model(mmm, method='advi')
```

## Dónde buscar más

- Para instalación detallada y ejemplos: `README.md`
- Para errores comunes y limpieza de caches: `TROUBLESHOOTING.md`
- Para ejemplos de insights listos para copiar en reportes: `INSIGHTS_EXAMPLES.md`
- Para historial de cambios: `CHANGELOG.md`

¡Listo — la app debe arrancar con los pasos anteriores!
from mmm_core import metrics
