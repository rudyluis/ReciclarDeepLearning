import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from keras.models import load_model
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
    return load_model("Garbage_clf.h5", compile=False, safe_mode=False)

model = load_model()

# --- Variables comunes ---
img_array = None

# --- Opción 1: Captura desde cámara móvil (st.camera_input) ---
st.markdown("### 📸 Captura desde cámara móvil")
img_camera = st.camera_input("Toma una foto")

if img_camera is not None:
    image = Image.open(img_camera)
    st.image(image, caption="Imagen capturada", use_column_width=True)
    img = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)

# --- Opción 2: Subida de imagen ---
if img_array is None:
    st.markdown("### 🖼️ O sube una imagen desde tu dispositivo")
    uploaded_file = st.file_uploader("Selecciona una imagen...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)
        img = image.resize(IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)

# --- Opción 3: Cámara web en PC (streamlit-webrtc) ---
if img_array is None:
    st.markdown("### 🎥 O usa la cámara web (en PC o laptop)")
    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.frame = None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            img = frame.to_ndarray(format="bgr24")
            self.frame = img
            return img

    camera_enabled = st.checkbox("Activar cámara web")

    if camera_enabled:
        webrtc_ctx = webrtc_streamer(key="webcam", video_transformer_factory=VideoTransformer)
        st.info("Haz una captura manual con el botón cuando la imagen esté lista.")

        if webrtc_ctx.video_transformer and webrtc_ctx.video_transformer.frame is not None:
            frame = webrtc_ctx.video_transformer.frame
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            st.image(img_pil, caption="Captura en vivo", use_column_width=True)

            if st.button("🧠 Predecir con captura de cámara web"):
                img = img_pil.resize(IMG_SIZE)
                img_array = tf.keras.preprocessing.image.img_to_array(img)

# --- Clasificación ---
if img_array is not None:
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    pred_class = labels[np.argmax(preds)]

    st.success(f"🔍 Residuo detectado: **{pred_class}**")
    st.write(f"📈 Probabilidad: **{preds[np.argmax(preds)]*100:.2f}%**")

    # --- Gráfico de barras ---
    fig, ax = plt.subplots()
    ax.bar(labels, preds, color='mediumseagreen')
    ax.set_ylabel('Probabilidad')
    ax.set_title('Distribución de predicción')
    plt.xticks(rotation=45)
    st.pyplot(fig)
