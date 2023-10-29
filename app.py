import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import cv2
import os
from tqdm import tqdm
from PIL import Image

predictor = joblib.load("mosquitos_v3.sav")

# Sidebar for navigation
st.sidebar.title("Opciones")
selection = st.sidebar.radio("Ir a", ["Realizar predicciones", "Performance de modelos", "Información extra"])

# Main content
if selection == "Realizar predicciones":
    st.title("Predicciones")

    # Upload an image for prediction
    uploaded_image = st.file_uploader("Cargar imagen", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Imagen cargada", use_column_width=True)
        
        # Button to Realizar predicciones
        if st.button("Realizar predicciones"):
            # Load and preprocess the uploaded image
            im = Image.open(uploaded_image)
            image = np.asarray(im)
            image = cv2.resize(image, (64, 64))  # Resize to (64, 64)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure images are in RGB format
            image = image / 255.0  # Normalize pixel values to [0, 1]
            image = np.array(image)

            # Make a prediction using the loaded model
            prediction = predictor.predict(image)  # Assuming predictor is a classifier

            # Display the prediction result and confidence scores
            st.write("Prediction:", prediction)

elif selection == "Performance de modelos":
    st.title("Performance de modelos")

    # Show performance metrics and interactive graphs
    0

elif selection == "Información extra":
    st.title("Información extra")

    # Display information about the dataset
    0