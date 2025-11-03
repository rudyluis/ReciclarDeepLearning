import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLBACKEND"] = "Agg"  # evita importar backends GUI

import streamlit as st
st.set_page_config(page_title="TF + Keras smoke", page_icon="🧪")

import keras
import tensorflow as tf
import numpy as np

st.title("TF/Keras/NumPy OK")
st.write("TF:", tf.__version__)
st.write("Keras:", keras.__version__)
st.write("NumPy:", np.__version__)