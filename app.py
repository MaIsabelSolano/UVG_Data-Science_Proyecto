'''
    Universidad del Valle de Guatemala
    Data Science 2023
    Grupo#5
'''

import io
from zipfile import ZipFile
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

def load_and_preprocess_images(image_paths, target_size):
    images = []
    x = 0
    for path in image_paths:
        x+=1
        image = cv2.imread("test_images/final/"+path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure images are in RGB format
        image = cv2.resize(image, target_size)
        image = image / 255.0  # Normalize pixel values to [0, 1]
        images.append(image)

    return np.array(images)

predictor = joblib.load("mosquitos_v3.sav")

# Sidebar for navigation
st.sidebar.title("Opciones")
selection = st.sidebar.radio("Ir a", ["Realizar predicciones", "Performance de modelos", "Información extra"])

# Main content
if selection == "Realizar predicciones":
    st.title("Predicciones")

    # Option to choose individual or batch predictions
    prediction_type = st.radio("Tipo de predicción", ["Imagen individual", "Conjunto de imágenes"])
    
    if prediction_type == "Imagen individual":
        # Upload an image for individual prediction
        uploaded_image = st.file_uploader("Cargar imagen", type=["jpg", "png", "jpeg"])
        
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Imagen cargada", use_column_width=True)
            
            # Button to Realizar predicciones para una sola imagen
            if st.button("Realizar predicciones"):
                # Load and preprocess the uploaded image
                image = Image.open(uploaded_image)
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure images are in RGB format
                image = cv2.resize(image, (128, 128))  # Resize to (128, 128)
                image = image / 255.0  # Normalize pixel values to [0, 1]

                # Make a prediction using the loaded model
                prediction = predictor.predict(np.array([image]))  # Assuming predictor is a classifier
                
                class_mapping = {
                    0: "Aedes aegypti",
                    1: "Aedes albopictus",
                    2: "Anopheles",
                    3: "Culex",
                    4: "Culiseta",
                    5: "Aedes japonicus/Aedes koreicus"
                }

                predicted_class = class_mapping[np.argmax(prediction[0])]
                
                # Display the predicted mosquito type
                st.markdown(f"<h3 style='color: blue;'>Tipo de mosquito: {predicted_class}</h3>", unsafe_allow_html=True)
                st.write(prediction)
            
                # Show information about the detected mosquito type
                if predicted_class == "Aedes aegypti":
                    st.subheader("Características:")
                    st.write("Este mosquito es de color oscuro con marcas blancas en su cuerpo y patas. Es conocido por ser portador del virus del dengue, la fiebre amarilla, el virus del Zika y el chikunguña.")
                    st.subheader("¿Qué contagia?:")
                    st.write("Transmite enfermedades como el dengue, fiebre amarilla, Zika y chikunguña.")
                    st.subheader("¿Qué hacer en caso de encontrarlo?:")
                    st.write("Elimina posibles criaderos de agua estancada en tu entorno, ya que estos mosquitos se reproducen en agua estancada.")
                    st.subheader("Estadísticas o datos interesantes:")
                    st.write("Aedes aegypti es responsable de propagar enfermedades virales graves en humanos.")
                    
                elif predicted_class == "Aedes albopictus":
                    st.subheader("Características:")
                    st.write("También conocido como mosquito tigre debido a sus rayas negras y blancas en el cuerpo y patas.")
                    st.subheader("¿Qué contagia?:")
                    st.write("Similar a Aedes aegypti, transmite el virus del dengue, Zika y chikunguña.")
                    st.subheader("¿Qué hacer en caso de encontrarlo?:")
                    st.write("Elimina criaderos de agua y usa repelentes para evitar picaduras.")
                    st.subheader("Estadísticas o datos interesantes:")
                    st.write("Se ha expandido a nivel mundial y es un vector importante de enfermedades.")
                    
                elif predicted_class == "Anopheles":
                    st.subheader("Características:")
                    st.write("Tienen cuerpos largos y delgados y alas moteadas. Son conocidos como vectores de la malaria.")
                    st.subheader("¿Qué contagia?:")
                    st.write("Transmite el parásito que causa la malaria en humanos.")
                    st.subheader("¿Qué hacer en caso de encontrarlo?:")
                    st.write("Usa mosquiteros y repelentes, y evita áreas donde son comunes.")
                    st.subheader("Estadísticas o datos interesantes:")
                    st.write("La malaria es una enfermedad grave que afecta a millones de personas en todo el mundo.")
                
                elif predicted_class == "Culex":
                    st.subheader("Características:")
                    st.write("Son mosquitos comunes con cuerpos delgados y patas finas. Suelen encontrarse en áreas urbanas y suburbanas.")
                    st.subheader("¿Qué contagia?:")
                    st.write("Pueden transmitir enfermedades como el virus del Nilo Occidental.")
                    st.subheader("¿Qué hacer en caso de encontrarlo?:")
                    st.write("Elimina agua estancada y usa repelentes.")
                    st.subheader("Estadísticas o datos interesantes:")
                    st.write("Pueden ser molestos debido a su picadura.")
                
                elif predicted_class == "Culiseta":
                    st.subheader("Características:")
                    st.write("Son mosquitos pequeños con alas angostas y patas delgadas.")
                    st.subheader("¿Qué contagia?:")
                    st.write("No suelen ser vectores de enfermedades significativas para los humanos.")
                    st.subheader("¿Qué hacer en caso de encontrarlo?:")
                    st.write("Controla su población si son una molestia.")
                    st.subheader("Estadísticas o datos interesantes:")
                    st.write("Su impacto en la salud humana es limitado.")
                
                elif predicted_class == "Aedes japonicus/Aedes koreicus":
                    st.subheader("Características:")
                    st.write("Estos dos tipos de mosquitos son difíciles de diferenciar y pertenecen a la misma especie compleja.")
                    st.subheader("¿Qué contagia?:")
                    st.write("Tienen el potencial de transmitir enfermedades como el virus del Zika.")
                    st.subheader("¿Qué hacer en caso de encontrarlo?:")
                    st.write("Monitorea y controla su población, ya que son relativamente nuevos en algunas regiones.")
                    st.subheader("Estadísticas o datos interesantes:")
                    st.write("Su distribución geográfica está en constante cambio.")
                    
    elif prediction_type == "Conjunto de imágenes":
        uploaded_zip = st.file_uploader("Cargar ZIP con imágenes", type=["zip"])
        
        if uploaded_zip is not None:
            st.write("Cargaste un archivo ZIP con imágenes. Realiza las predicciones para las imágenes en el grupo.")

            # Load and preprocess images from the ZIP file
            with ZipFile(uploaded_zip, 'r') as zip_file:
                image_paths = zip_file.namelist()
                grouped_images = {}  # Inicializa un diccionario para agrupar imágenes por tipo de mosquito
                uploaded_images = []
                
                for image_path in image_paths:
                    try:
                        # Load and preprocess each image
                        image_bytes = zip_file.read(image_path)
                        image = Image.open(io.BytesIO(image_bytes))
                        uploaded_images.append((image, image_path))
                        image = np.array(image)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure images are in RGB format
                        image = cv2.resize(image, (128, 128))  # Resize to (128, 128)
                        image = image / 255.0  # Normalize pixel values to [0, 1]

                        # Make a prediction using the loaded model
                        prediction = predictor.predict(np.array([image]))  # Assuming predictor is a classifier

                        class_mapping = {
                            0: "Aedes aegypti",
                            1: "Aedes albopictus",
                            2: "Anopheles",
                            3: "Culex",
                            4: "Culiseta",
                            5: "Aedes japonicus/Aedes koreicus"
                        }

                        predicted_class = class_mapping[np.argmax(prediction)]

                        # Asigna la imagen al tipo de mosquito con alta confianza
                        if predicted_class not in grouped_images:
                            grouped_images[predicted_class] = []
                        grouped_images[predicted_class].append(image_path)
                        

                    except Exception as e:
                        st.write(f"Error al procesar la imagen {image_path}: {str(e)}")

            # Display uploaded images with their names as captions
            st.subheader("Imágenes subidas:")
            columns = 4  # Puedes ajustar la cantidad de columnas en el grid
            for i in range(0, len(uploaded_images), columns):
                images_row = uploaded_images[i:i + columns]
                st.image([image for image, _ in images_row], caption=[name for _, name in images_row], width=150)

            # Display information and types for each group of mosquito images
            for mosquito_type, image_paths in grouped_images.items():
                st.markdown(f"<h3 style='color: blue;'>Tipo de mosquito: {mosquito_type}</h3>", unsafe_allow_html=True)
                st.write("Imágenes en este grupo:")
                for image_path in image_paths:
                    st.write(image_path)
                    
                # Show information about the detected mosquito type
                if mosquito_type == "Aedes aegypti":
                    st.subheader("Características:")
                    st.write("Este mosquito es de color oscuro con marcas blancas en su cuerpo y patas. Es conocido por ser portador del virus del dengue, la fiebre amarilla, el virus del Zika y el chikunguña.")
                    st.subheader("¿Qué contagia?:")
                    st.write("Transmite enfermedades como el dengue, fiebre amarilla, Zika y chikunguña.")
                    st.subheader("¿Qué hacer en caso de encontrarlo?:")
                    st.write("Elimina posibles criaderos de agua estancada en tu entorno, ya que estos mosquitos se reproducen en agua estancada.")
                    st.subheader("Estadísticas o datos interesantes:")
                    st.write("Aedes aegypti es responsable de propagar enfermedades virales graves en humanos.")
                    
                elif mosquito_type == "Aedes albopictus":
                    st.subheader("Características:")
                    st.write("También conocido como mosquito tigre debido a sus rayas negras y blancas en el cuerpo y patas.")
                    st.subheader("¿Qué contagia?:")
                    st.write("Similar a Aedes aegypti, transmite el virus del dengue, Zika y chikunguña.")
                    st.subheader("¿Qué hacer en caso de encontrarlo?:")
                    st.write("Elimina criaderos de agua y usa repelentes para evitar picaduras.")
                    st.subheader("Estadísticas o datos interesantes:")
                    st.write("Se ha expandido a nivel mundial y es un vector importante de enfermedades.")
                    
                elif mosquito_type == "Anopheles":
                    st.subheader("Características:")
                    st.write("Tienen cuerpos largos y delgados y alas moteadas. Son conocidos como vectores de la malaria.")
                    st.subheader("¿Qué contagia?:")
                    st.write("Transmite el parásito que causa la malaria en humanos.")
                    st.subheader("¿Qué hacer en caso de encontrarlo?:")
                    st.write("Usa mosquiteros y repelentes, y evita áreas donde son comunes.")
                    st.subheader("Estadísticas o datos interesantes:")
                    st.write("La malaria es una enfermedad grave que afecta a millones de personas en todo el mundo.")
                
                elif mosquito_type == "Culex":
                    st.subheader("Características:")
                    st.write("Son mosquitos comunes con cuerpos delgados y patas finas. Suelen encontrarse en áreas urbanas y suburbanas.")
                    st.subheader("¿Qué contagia?:")
                    st.write("Pueden transmitir enfermedades como el virus del Nilo Occidental.")
                    st.subheader("¿Qué hacer en caso de encontrarlo?:")
                    st.write("Elimina agua estancada y usa repelentes.")
                    st.subheader("Estadísticas o datos interesantes:")
                    st.write("Pueden ser molestos debido a su picadura.")
                
                elif mosquito_type == "Culiseta":
                    st.subheader("Características:")
                    st.write("Son mosquitos pequeños con alas angostas y patas delgadas.")
                    st.subheader("¿Qué contagia?:")
                    st.write("No suelen ser vectores de enfermedades significativas para los humanos.")
                    st.subheader("¿Qué hacer en caso de encontrarlo?:")
                    st.write("Controla su población si son una molestia.")
                    st.subheader("Estadísticas o datos interesantes:")
                    st.write("Su impacto en la salud humana es limitado.")
                
                elif mosquito_type == "Aedes japonicus/Aedes koreicus":
                    st.subheader("Características:")
                    st.write("Estos dos tipos de mosquitos son difíciles de diferenciar y pertenecen a la misma especie compleja.")
                    st.subheader("¿Qué contagia?:")
                    st.write("Tienen el potencial de transmitir enfermedades como el virus del Zika.")
                    st.subheader("¿Qué hacer en caso de encontrarlo?:")
                    st.write("Monitorea y controla su población, ya que son relativamente nuevos en algunas regiones.")
                    st.subheader("Estadísticas o datos interesantes:")
                    st.write("Su distribución geográfica está en constante cambio.")
        

elif selection == "Performance de modelos":
    st.title("Performance de modelos")

    # Show performance metrics and interactive graphs
    0

elif selection == "Información extra":
    st.title("Información extra")

    # Display information about the dataset
    0