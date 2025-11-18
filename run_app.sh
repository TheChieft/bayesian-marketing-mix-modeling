#!/bin/bash
# Script para ejecutar la aplicación MMM con el entorno conda correcto

echo "🚀 Iniciando Marketing Mix Modeling Dashboard..."
echo ""

# Limpiar cache de Streamlit y Python
echo "🔄 Limpiando cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
if [ -d ~/.streamlit/cache ]; then
    rm -rf ~/.streamlit/cache/* 2>/dev/null
fi
echo "✅ Cache limpiado"
echo ""

# Activar entorno conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mmm_bayes

# Verificar que el entorno esté activo
if [ $? -eq 0 ]; then
    echo "✅ Entorno 'mmm_bayes' activado"
    echo ""
    
    # Ejecutar aplicación Streamlit
    streamlit run app/app_mmm_streamlit.py
else
    echo "❌ Error: No se pudo activar el entorno 'mmm_bayes'"
    echo "Por favor, crea el entorno con: conda create -n mmm_bayes python=3.10"
    echo "E instala las dependencias: pip install -r requirements.txt"
    exit 1
fi
