import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import os
tf.config.set_visible_devices([], 'GPU')
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # menos logs
# --- Configuración ---
st.set_page_config(page_title="Clasificador de Residuos", layout="centered")
st.title("♻️ Clasificador de Residuos con IA")
st.markdown("Puedes tomar una foto desde tu **cámara móvil o web**, o subir una imagen para predecir el tipo de residuo.")

# Etiquetas en español
labels = ['Vidrio', 'Papel', 'Plástico', 'Cartón', 'Metal', 'Orgánico/Basura']
IMG_SIZE = (160, 160)

# --- Cargar modelo ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Garbage_clf.h5', compile=False)

model = load_model()
