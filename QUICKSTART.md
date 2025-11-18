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

**Nuevas características UX (Fase 3 & 4):**
- 📁 **Selector de dataset**: Elige entre ejemplo incluido o subir tu propio CSV
- ✅ **Validación automática**: Verifica que tu CSV tenga el formato correcto
- 📊 **Escala de unidades**: Muestra valores en unidades originales, miles o millones
- 💡 **Insights automáticos**: Análisis de negocio generado automáticamente
- 📥 **Reporte descargable**: Descarga todo el análisis en formato Markdown

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

# Visualizar
fig = viz.plot_actual_vs_predicted(y, y_pred)
fig.show()
```

### Opción 3: Ejecutar Ejemplo

```bash
conda activate mmm_bayes
python example_usage.py
```

### Opción 4: Análisis Exploratorio

```bash
conda activate mmm_bayes
jupyter notebook notebooks/01_eda_mmm.ipynb
```

## Estructura de mmm_core

| Módulo | Función Principal |
|--------|-------------------|
| `data.py` | Carga, validación, sanitización |
| `transforms.py` | Adstock, Hill, estandarización |
| `model.py` | PyMC: construcción, ajuste, predicción |
| `metrics.py` | R², RMSE, MAPE, ROI, ROAS, contribuciones |
| `viz.py` | Gráficos con Plotly |

## Parámetros Clave

### Adstock
- **Tasa (r)**: 0.0 - 0.9
- Efecto: Modela persistencia del impacto publicitario
- Recomendado: 0.1 - 0.3

### Hill (Saturación)
- **Gamma (γ)**: 0.5 - 3.0
- Efecto: Modela rendimientos decrecientes
- Recomendado: 1.0 - 2.0

### Método de Inferencia
- **ADVI**: Rápido (~segundos), aproximado
- **NUTS**: Lento (~minutos), más preciso

## Métricas de Salida

- **R²**: Bondad de ajuste (0-1, mayor es mejor)
- **RMSE**: Error cuadrático medio (menor es mejor)
- **MAPE**: Error porcentual absoluto medio (menor es mejor)
- **ROI**: (Contribución - Inversión) / Inversión
- **ROAS**: Contribución / Inversión (revenue per dollar)
- **Share of Sales**: Contribución / Ventas Totales

## Troubleshooting

### Error: ModuleNotFoundError
```bash
# Asegúrate de tener el entorno activado
conda activate mmm_bayes
pip install -r requirements.txt
```

### Error: FileNotFoundError
```bash
# Verifica que estés en el directorio correcto
cd /path/to/MarketingMixModeling
```

### Advertencia: ArviZ shape validation
Es normal con ADVI (1 chain). Para múltiples chains usa NUTS.

## Recursos

- 📚 [Documentación PyMC](https://www.pymc.io/)
- 📖 [Paper: Bayesian MMM](https://www.pymc-labs.io/blog-posts/mmm-google/)
- 🎥 [Tutorial MMM](https://www.youtube.com/results?search_query=marketing+mix+modeling+pymc)

## Soporte

Para preguntas o issues: [GitHub Issues](https://github.com/TheChieft/bayesian-marketing-mix-modeling/issues)

---

✨ **Happy Modeling!**
