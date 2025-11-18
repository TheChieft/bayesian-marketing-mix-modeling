"""
Script de ejemplo mostrando cómo usar mmm_core independientemente de Streamlit.
"""

from mmm_core import data, transforms, model, metrics
import numpy as np

print("=" * 60)
print("Ejemplo de uso de mmm_core (sin Streamlit)")
print("=" * 60)

# 1. Generar datos sintéticos
print("\n1️⃣ Generando datos sintéticos...")
df = data.generate_synthetic_data(n=100, seed=42)
df, mapping = data.sanitize_columns(df)
print(f"   ✅ {df.shape[0]} observaciones generadas")

# 2. Identificar columnas
media_cols = [c for c in df.columns if 'TV' in c or 'Radio' in c or 'Newspaper' in c]
sales_col = [c for c in df.columns if 'Sales' in c][0]
print(f"   📊 Variables de medios: {media_cols}")
print(f"   💰 Variable objetivo: {sales_col}")

# 3. Aplicar transformaciones
print("\n2️⃣ Aplicando transformaciones (Adstock + Hill)...")
df_trans, sat_cols = transforms.build_transformed_media(
    df, media_cols, 
    adstock_rate=0.1, 
    hill_gamma=1.5
)
print(f"   ✅ Columnas transformadas: {sat_cols}")

# 4. Preparar datos
print("\n3️⃣ Preparando datos para el modelo...")
X = df_trans[sat_cols].values
y = df_trans[sales_col].values
X_scaled, y_scaled, scaler_X, scaler_y = transforms.standardize_data(X, y)
print(f"   ✅ X: {X_scaled.shape}, y: {y_scaled.shape}")

# 5. Construir y ajustar modelo
print("\n4️⃣ Construyendo y ajustando modelo bayesiano (ADVI)...")
mmm = model.build_mmm_model(X_scaled, y_scaled)
idata = model.fit_mmm_model(mmm, method='advi', n_advi=5000)
print("   ✅ Modelo ajustado")

# 6. Obtener coeficientes
print("\n5️⃣ Extrayendo resultados...")
summary = model.get_posterior_summary(idata)
beta_means = model.extract_beta_coefficients(idata)
alpha_mean = summary.loc["alpha", "mean"]

print(f"   α (baseline): {alpha_mean:.4f}")
for i, col in enumerate(media_cols):
    print(f"   β_{col}: {beta_means[i]:.4f}")

# 7. Predicciones
print("\n6️⃣ Generando predicciones...")
mmm_pred = model.build_mmm_model(X_scaled, y_scaled)
y_pred_scaled, _ = model.predict_posterior(mmm_pred, idata, X_scaled)
y_pred = transforms.inverse_transform_predictions(y_pred_scaled, scaler_y)

# 8. Métricas
print("\n7️⃣ Calculando métricas de ajuste...")
fit_metrics = metrics.compute_fit_metrics(y, y_pred)
print(f"   R²:   {fit_metrics['R2']:.3f}")
print(f"   RMSE: {fit_metrics['RMSE']:.3f}")
print(f"   MAPE: {fit_metrics['MAPE']:.2f}%")

# 9. Contribuciones
print("\n8️⃣ Calculando contribuciones por canal...")
contrib_df, baseline, contributions = metrics.compute_contributions(
    X, beta_means, alpha_mean, scaler_X, scaler_y, media_cols
)

total_sales = float(y.sum())
print(f"\n   Total ventas: ${total_sales:,.2f}")
print(f"   Baseline: ${baseline:,.2f} ({baseline/total_sales*100:.1f}%)")
for channel in media_cols:
    contrib = contributions[channel]
    pct = contrib / total_sales * 100
    print(f"   {channel}: ${contrib:,.2f} ({pct:.1f}%)")

# 10. ROI/ROAS
print("\n9️⃣ Calculando ROI y ROAS...")
contrib_df = metrics.compute_roi_roas(contrib_df, df, media_cols, total_sales)
print("\n", contrib_df[['Canal', 'ROI', 'ROAS', 'Share_of_Sales']].to_string(index=False))

print("\n" + "=" * 60)
print("✅ Ejemplo completado exitosamente")
print("=" * 60)
