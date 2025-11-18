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
| `data.py` | Carga, validación, sanitización, schema checking |
| `transforms.py` | Adstock, Hill, estandarización |
| `model.py` | PyMC: construcción, ajuste, predicción, train/test split |
| `metrics.py` | R², RMSE, MAPE, ROI, ROAS, contribuciones, intervalos de credibilidad |
| `viz.py` | Gráficos con Plotly (incluye diagnósticos) |

## Nuevas funciones - Fase 5

### Train/Test Split

```python
from mmm_core import model

# División temporal sin shuffle
X_train, X_test, y_train, y_test = model.split_train_test(
    X_scaled, y_scaled, test_size=0.3, shuffle=False
)

# Fit con métricas de validación
mmm, idata, metrics_dict = model.fit_mmm_with_validation(
    X_train, y_train, X_test, y_test, method='advi'
)

print(f"Train R²: {metrics_dict['train_r2']:.3f}")
print(f"Test R²: {metrics_dict['test_r2']:.3f}")
```

### Intervalos de Credibilidad

```python
from mmm_core import metrics

# Calcular IC 90% (5-95 percentiles)
uncertainty_df = metrics.compute_contribution_uncertainty(
    X_saturated, idata, scaler_X, scaler_y, media_cols, ci_level=0.90
)

# uncertainty_df contiene:
# - Contribución_media: valor esperado
# - CI_lower: percentil 5
# - CI_upper: percentil 95
# - CI_width: ancho del intervalo

# Interpretar
for _, row in uncertainty_df.iterrows():
    print(f"{row['Canal']}: {row['Contribución_media']:.0f} "
          f"[{row['CI_lower']:.0f}, {row['CI_upper']:.0f}]")
```

### Diagnósticos Estadísticos

```python
from mmm_core import viz

residuals = y_true - y_pred

# 1. Residuos vs Predicción
fig1 = viz.plot_residuals_vs_predicted(y_pred, residuals)
fig1.show()

# 2. Distribución de residuos
fig2 = viz.plot_residuals_histogram(residuals)
fig2.show()

# 3. Q-Q Plot
fig3 = viz.plot_qq_plot(residuals)
fig3.show()
```

**Interpretación:**
- Residuos dispersos aleatoriamente → buen ajuste ✓
- Histograma con forma gaussiana → supuestos cumplidos ✓
- Puntos en Q-Q plot sobre línea diagonal → normalidad ✓

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
