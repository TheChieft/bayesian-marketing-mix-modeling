# 🔧 Solución de Problemas - Fase 5

## Problema: AttributeError con funciones nuevas

Si ves errores como:
```
AttributeError: module 'mmm_core.metrics' has no attribute 'compute_contribution_uncertainty'
AttributeError: module 'mmm_core.model' has no attribute 'split_train_test'
```

### Solución Rápida (Opción 1 - Recomendada)

**El problema es que Streamlit cachea los módulos de Python.**

Simplemente ejecuta:
```bash
./run_app.sh
```

El script ahora limpia automáticamente el cache antes de iniciar. Si ya estaba ejecutando la app:

1. **Detén la app** (Ctrl+C en la terminal)
2. **Ejecuta el script actualizado:**
   ```bash
   ./run_app.sh
   ```

### Solución Manual (Opción 2)

Si aún así ves el error:

```bash
# 1. Detén Streamlit (Ctrl+C)

# 2. Limpia el cache de Python
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# 3. Limpia el cache de Streamlit
rm -rf ~/.streamlit/cache/*

# 4. Reinicia la app
conda activate mmm_bayes
streamlit run app/app_mmm_streamlit.py
```

### Solución Nuclear (Opción 3)

Si ninguna de las anteriores funciona:

```bash
# Limpiar COMPLETAMENTE y reinstalar
conda deactivate
rm -rf ~/.streamlit/cache
find . -type d -name __pycache__ -delete
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type d -name ".pytest_cache" -delete

# Reinstalar el entorno
conda deactivate
conda remove -n mmm_bayes --all
conda create -n mmm_bayes python=3.10
conda activate mmm_bayes
pip install -r requirements.txt

# Ejecutar
./run_app.sh
```

---

## Verificación rápida de imports

Para confirmar que las funciones están disponibles, ejecuta:

```bash
conda activate mmm_bayes
python -c "
from mmm_core import model, metrics, viz
print('✓ split_train_test:', hasattr(model, 'split_train_test'))
print('✓ fit_mmm_with_validation:', hasattr(model, 'fit_mmm_with_validation'))
print('✓ compute_contribution_uncertainty:', hasattr(metrics, 'compute_contribution_uncertainty'))
print('✓ compute_baseline_uncertainty:', hasattr(metrics, 'compute_baseline_uncertainty'))
print('✓ plot_residuals_vs_predicted:', hasattr(viz, 'plot_residuals_vs_predicted'))
print('✓ plot_residuals_histogram:', hasattr(viz, 'plot_residuals_histogram'))
print('✓ plot_qq_plot:', hasattr(viz, 'plot_qq_plot'))
"
```

Si todos muestran `True`, las funciones están correctamente instaladas.

---

## Notas importantes

- **No hagas cambios en los archivos mientras Streamlit está corriendo**
- Si necesitas cambios en `mmm_core/`, siempre reinicia la app
- El script `run_app.sh` ahora limpia automáticamente el cache
- Para desarrollo, considera usar `streamlit run ... --logger.level=debug`

---

## ¿Todavía hay problemas?

1. Verifica que estés en el directorio correcto:
   ```bash
   pwd  # Debe terminar en: .../MarketingMixModeling
   ```

2. Verifica el entorno:
   ```bash
   conda info  # Busca 'mmm_bayes' en active env o envs
   ```

3. Verifica los archivos existen:
   ```bash
   ls -la mmm_core/model.py mmm_core/metrics.py mmm_core/viz.py
   ```

Si todo falla, abre un issue con:
- Output completo del error
- Python version: `python --version`
- Conda: `conda --version`
