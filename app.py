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

# Título de la aplicación
st.title('Mosquitos')

# Sidebar for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Make Predictions", "Model Performance", "Data Information"])

# Main content
if selection == "Make Predictions":
    st.title("Model Predictions")

    # Upload an image for prediction
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Button to make predictions
        if st.button("Make Predictions"):
            # Load and preprocess the uploaded image
            image = cv2.imread(uploaded_image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure images are in RGB format
            image = image / 255.0  # Normalize pixel values to [0, 1]
            image = cv2.resize(image, (64, 64))  # Resize to (64, 64)
            image = np.array(image)

            # Make a prediction using the loaded model
            prediction = predictor.predict([image])  # Assuming predictor is a classifier
            confidence_scores = predictor.predict_proba([image])  # Get confidence scores

            # Display the prediction result and confidence scores
            st.write("Prediction:", prediction)
            st.write("Confidence Scores:", confidence_scores)

elif selection == "Model Performance":
    st.title("Model Performance")

    # Show performance metrics and interactive graphs
    0

elif selection == "Data Information":
    st.title("Data Information")

    # Display information about the dataset
    0