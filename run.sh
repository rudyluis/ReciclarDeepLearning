#!/bin/bash
# Script para ejecutar tu app Streamlit con TF sin conflictos

# 1. Crear/activar entorno virtual
if [ ! -d "venv" ]; then
    python3.10 -m venv venv
fi
source venv/bin/activate

# 2. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 3. Aumentar límite de watchers inotify
sudo sysctl fs.inotify.max_user_watches=524288
sudo sysctl -p

# 4. Desactivar watchdog de Streamlit
export STREAMLIT_WATCHDOG=false

# 5. Ejecutar app
streamlit run app.py --server.runOnSave false
