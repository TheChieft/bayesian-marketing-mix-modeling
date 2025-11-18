#!/bin/bash

# Script para limpiar el cache de Streamlit y reiniciar la app
# Útil cuando hay cambios en los módulos que no se cargan correctamente

echo "🔄 Limpiando cache de Streamlit..."

# Limpiar directorio de cache
if [ -d ~/.streamlit/cache ]; then
    rm -rf ~/.streamlit/cache/*
    echo "✅ Cache de Streamlit limpiado"
fi

# Limpiar pycache de Python
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
echo "✅ Cache de Python (__pycache__) limpiado"

# Limpiar archivos .pyc
find . -type f -name "*.pyc" -delete 2>/dev/null
echo "✅ Archivos .pyc eliminados"

echo ""
echo "🚀 Iniciando Streamlit con módulos frescos..."
echo "═════════════════════════════════════════════"
echo ""

# Activar entorno conda y lanzar app
if command -v conda &> /dev/null; then
    # Asumiendo que estamos en el entorno correcto
    streamlit run app/app_mmm_streamlit.py --logger.level=debug
else
    echo "⚠️ Conda no encontrada. Lanzando directamente..."
    streamlit run app/app_mmm_streamlit.py --logger.level=debug
fi
