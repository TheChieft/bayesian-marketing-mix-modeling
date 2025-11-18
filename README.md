# Marketing Mix Modeling con PyMC

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyMC](https://img.shields.io/badge/PyMC-5.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Un proyecto académico de **Marketing Mix Modeling (MMM) Bayesiano** construido con PyMC, diseñado para analizar el impacto de diferentes canales de marketing en las ventas y optimizar la inversión publicitaria.

## 🎯 Características

- **Modelado Bayesiano**: Utiliza PyMC para inferencia probabilística robusta
- **Transformaciones avanzadas**: 
  - Adstock para efectos de arrastre
  - Saturación Hill para rendimientos decrecientes
- **Métricas completas**: ROI, ROAS, contribuciones por canal
- **Intervalos de credibilidad**: Cuantificación de incertidumbre (IC 90%)
- **Train/Test split**: Validación rigurosa con métricas in-sample y out-of-sample
- **Diagnósticos estadísticos**: Residuos, Q-Q plots, heteroscedasticidad
- **Opciones de inferencia**: ADVI (rápido) y NUTS (preciso)
- **Visualizaciones interactivas**: Gráficos de contribución, cascada, y comparación real vs predicho
- **Insights automáticos**: Análisis de negocio generado por IA
- **Arquitectura modular**: Código reutilizable y mantenible

## 📁 Estructura del proyecto

```
MarketingMixModeling/
├── app/
│   └── app_mmm_streamlit.py      # Dashboard Streamlit
├── mmm_core/                      # Biblioteca core reutilizable
│   ├── __init__.py
│   ├── data.py                    # Carga y validación de datos
│   ├── transforms.py              # Adstock, Hill, estandarización
│   ├── model.py                   # Construcción y ajuste del modelo PyMC
│   ├── metrics.py                 # R², RMSE, MAPE, ROI, ROAS
│   └── viz.py                     # Funciones de visualización
├── data/
│   └── Basemediosfinal.csv        # Datos de ejemplo
├── notebooks/
│   └── 01_eda_mmm.ipynb           # Análisis exploratorio
├── requirements.txt               # Dependencias
├── LICENSE
└── README.md
```

## 🚀 Instalación

### Prerequisitos

- Python 3.8 o superior
- pip o conda

### Pasos

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/TheChieft/bayesian-marketing-mix-modeling.git
   cd bayesian-marketing-mix-modeling
   ```

2. **Crear entorno virtual (recomendado)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Uso

### Ejecutar la aplicación Streamlit

```bash
streamlit run app/app_mmm_streamlit.py
```

La aplicación estará disponible en `http://localhost:8501`

### Usar la biblioteca mmm_core en Python

```python
from mmm_core import data, transforms, model, metrics, viz

# Cargar datos
df = data.load_base_data("data/Basemediosfinal.csv")
df, mapping = data.sanitize_columns(df)

# Aplicar transformaciones
df_trans, sat_cols = transforms.build_transformed_media(
    df, 
    media_cols=["TV", "Radio", "Newspaper"],
    adstock_rate=0.1,
    hill_gamma=1.5
)

# Preparar datos
X = df_trans[sat_cols].values
y = df_trans["Sales"].values
X_scaled, y_scaled, scaler_X, scaler_y = transforms.standardize_data(X, y)

# Construir y ajustar modelo
mmm = model.build_mmm_model(X_scaled, y_scaled)
idata = model.fit_mmm_model(mmm, method='advi')

# Calcular métricas
beta_means = model.extract_beta_coefficients(idata)
contrib_df, baseline, contributions = metrics.compute_contributions(
    X, beta_means, alpha_mean, scaler_X, scaler_y, media_cols
)

# Visualizar
fig = viz.plot_contribution_pie(media_cols, contributions, baseline, 0)
fig.show()
```

## 📊 Metodología

### 1. Transformaciones de medios

#### Adstock
Modela el efecto de arrastre de la publicidad:
```
y[t] = x[t] + r * y[t-1]
```
donde `r` es la tasa de decaimiento (0-1)

#### Saturación Hill
Modela rendimientos decrecientes:
```
f(x) = α * x^γ / (θ^γ + x^γ)
```
- `α`: Nivel de saturación máximo
- `θ`: Punto de semi-saturación
- `γ`: Forma de la curva (>1 es S-shaped)

### 2. Modelo Bayesiano

```
α ~ Normal(0, 5)                    # Baseline
β_i ~ TruncatedNormal(0, 5, lower=0) # Coeficientes (≥0)
σ ~ HalfNormal(2)                   # Error estándar

μ = α + Σ(β_i * X_i)
y ~ StudentT(ν=5, μ=μ, σ=σ)         # Likelihood robusto
```

### 3. Métricas de negocio

- **ROI** (Return on Investment): `(Contribución - Inversión) / Inversión`
- **ROAS** (Return on Ad Spend): `Contribución / Inversión`
- **Share of Sales**: `Contribución / Ventas totales`

### 4. Intervalos de Credibilidad (Fase 5)

Cuantificación de incertidumbre usando el posterior bayesiano:

```python
from mmm_core import metrics

# Calcular IC 90% (5-95 percentiles)
uncertainty_df = metrics.compute_contribution_uncertainty(
    X_saturated, idata, scaler_X, scaler_y, media_cols, ci_level=0.90
)
# Retorna: Canal, Contribución_media, CI_lower, CI_upper, CI_width
```

**Interpretación:**
- IC estrecho (CI_width pequeño) = alta certeza
- IC amplio = alta incertidumbre, recolectar más datos
- Ejemplo: "TV aporta entre 30-40% de ventas con 90% confianza"

### 5. Train/Test Split (Fase 5)

Validación rigurosa para detectar overfitting:

```python
from mmm_core import model

# Split temporal (respeta orden cronológico)
X_train, X_test, y_train, y_test = model.split_train_test(
    X_scaled, y_scaled, test_size=0.3, shuffle=False
)

# Fit con validación automática
mmm, idata, metrics_dict = model.fit_mmm_with_validation(
    X_train, y_train, X_test, y_test, method='advi'
)

# metrics_dict contiene: train_r2, train_rmse, test_r2, test_rmse, etc.
```

**Criterios de overfitting:**
- Si |R²_train - R²_test| > 0.15 → Overfitting severo
- Si |R²_train - R²_test| < 0.05 → Buen ajuste

### 6. Diagnósticos Estadísticos (Fase 5)

Verificación de supuestos del modelo:

```python
from mmm_core import viz

residuals = y_true - y_pred

# 1. Residuos vs Predicción (heteroscedasticidad)
fig1 = viz.plot_residuals_vs_predicted(y_pred, residuals)

# 2. Histograma de Residuos (normalidad)
fig2 = viz.plot_residuals_histogram(residuals)

# 3. Q-Q Plot (normalidad)
fig3 = viz.plot_qq_plot(residuals)
```

**Qué buscar:**
- Residuos dispersos aleatoriamente alrededor de 0 ✓
- Histograma con forma de campana gaussiana ✓
- Puntos del Q-Q plot siguiendo la línea diagonal ✓

## 🧪 Testing

Para ejecutar tests (cuando estén implementados):
```bash
pytest tests/
```

## 📚 Recursos adicionales

- [PyMC Documentation](https://www.pymc.io/)
- [Marketing Mix Modeling Guide](https://en.wikipedia.org/wiki/Marketing_mix_modeling)
- [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)

## 🤝 Contribuciones

Este es un proyecto académico, pero las sugerencias y mejoras son bienvenidas:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👨‍💻 Autor

**TheChieft**
- GitHub: [@TheChieft](https://github.com/TheChieft)
- Repository: [bayesian-marketing-mix-modeling](https://github.com/TheChieft/bayesian-marketing-mix-modeling)

## 🙏 Agradecimientos

- Equipo de PyMC por la excelente biblioteca
- Comunidad académica DeFi
- Recursos de la comunidad de Data Science

---

⭐ Si este proyecto te resultó útil, considera darle una estrella en GitHub!
