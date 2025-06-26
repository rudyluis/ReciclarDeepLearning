import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# --- Configuración ---
st.set_page_config(page_title="Clasificador de Residuos", layout="centered")

# Etiquetas en español
labels = ['Cartón', 'Vidrio', 'Metal', 'Papel', 'Plástico', 'Orgánico/Basura']
IMG_SIZE = (160, 160)

# Cargar modelo
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Garbage_clf.h5', compile=False)

model = load_model()

st.title("♻️ Clasificador de Residuos con IA")
st.markdown("Sube una imagen o usa tu cámara para predecir el tipo de residuo.")

# --- Captura desde cámara ---
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        return img

camera_enabled = st.checkbox("📷 Usar cámara web")

img_array = None

if camera_enabled:
    webrtc_ctx = webrtc_streamer(key="webcam", video_transformer_factory=VideoTransformer)
    st.info("Activa la cámara y haz una captura (screenshot manual por ahora).")

    if webrtc_ctx.video_transformer and webrtc_ctx.video_transformer.frame is not None:
        frame = webrtc_ctx.video_transformer.frame
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        st.image(img_pil, caption="Captura en vivo", use_column_width=True)

        if st.button("🧠 Predecir con captura de cámara"):
            img = img_pil.resize(IMG_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
else:
    uploaded_file = st.file_uploader("Sube una imagen...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)
        img = image.resize(IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)

# --- Clasificación ---
if img_array is not None:
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    pred_class = labels[np.argmax(preds)]

    st.success(f"🔍 Residuo detectado: **{pred_class}**")
    st.write(f"📈 Probabilidad: **{preds[np.argmax(preds)]*100:.2f}%**")
    fig, ax = plt.subplots()
    ax.bar(labels, preds, color='lightgreen')
    ax.set_ylabel('Probabilidad')
    ax.set_title('Distribución de predicción')
    plt.xticks(rotation=45)
    st.pyplot(fig)


