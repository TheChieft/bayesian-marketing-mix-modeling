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
# 🔧 Solución de Problemas - (resumen y pasos rápidos)

Este documento contiene recetas para los problemas más comunes al ejecutar la app.

Si necesitas instrucciones más largas o contexto técnico, ve a `README.md` o al `QUICKSTART.md`.

## Problemas frecuentes

- AttributeError por funciones faltantes en `mmm_core` (p.ej. `split_train_test`) → suele ser caché de Streamlit o Python.
- ModuleNotFoundError → dependencias no instaladas.
- FileNotFoundError → ruta de trabajo incorrecta o datos faltantes.

## Pasos rápidos (recomendado)

1. Detén la app (Ctrl+C).
2. Ejecuta el script que limpia caches y lanza la app:

```bash
./run_app.sh
```

Si eso no resuelve, usa la limpieza manual:

```bash
# Limpia caches de Python y Streamlit
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
rm -rf ~/.streamlit/cache/*

# Reinicia la app en tu entorno
conda activate mmm_bayes
streamlit run app/app_mmm_streamlit.py
```

## Dependencias faltantes

Si ves `ModuleNotFoundError`, instala las dependencias:

```bash
pip install -r requirements.txt
```

## Verificación rápida de availability de funciones

```bash
python -c "from mmm_core import model, metrics, viz; print('split:', hasattr(model,'split_train_test'))"
```

## Si todo falla

- Revisa los logs de Streamlit (en local o en Streamlit Cloud -> Manage app → Logs).
- Abre un issue con el traceback completo y la versión de Python/conda.

---

Enlaces rápidos: `QUICKSTART.md` (instalación y ejecución), `README.md` (documentación principal).
print('✓ plot_qq_plot:', hasattr(viz, 'plot_qq_plot'))
