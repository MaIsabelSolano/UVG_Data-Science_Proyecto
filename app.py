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

# predictor = joblib.load("mosquitos_v3.sav")
predictor = joblib.load("extra.sav")

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
                st.write(f"Imágenes en este grupo: {image_paths}")
                    
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
    model_to_use = st.radio("Modelo a utilizar", ["1 (Simple)", "2 (Second training)", "3 (80%)"])
    model_images = ["model1.jpg", "model2.jpg", "model3.jpg"]
    if model_to_use == "1 (Simple)":
        
        
        st.markdown("Precisión")
        df_1 = pd.read_csv('./comparison/A.csv')

        # Seleccionar las columnas de interés
        df_selected = df_1[['Training Loss', 'Validation Loss']]

        # Mostrar el gráfico de líneas
        st.line_chart(data=df_selected)
        
        st.markdown("Perdida")
        
        df_selected = df_1[['Training Accuracy','Validation Accuracy']]

        # Mostrar el gráfico de líneas
        st.line_chart(data=df_selected)
        
        
        # st.image(f"./comparison/{model_images[0]}", use_column_width=True, caption="mosquitos_v1")
        st.subheader("ACCURACY: 60.08%")
        st.write("Este corresponde a unmodelo de red neuronal convolucional (CNN) utilizando TensorFlow y Keras. Este modelo se enfoca en el procesamiento de imágenes de mosquitos Y Se compone de capas convolucionales para la extracción de características, seguidas de capas de Max Pooling para reducir la dimensionalidad. Tras aplanar la salida, se incluyen capas completamente conectadas con una capa de dropout para regularización. Finalmente, se agrega una capa de salida con una activación softmax para la clasificación. El accuracy que posee es relativamente bajo debido a que este aproach busca la familiarizacion con el dataset a si mismo que un aproach sencillo inicial")
    if model_to_use == "2 (Second training)":
        
        st.markdown("Precisión")
        df_2 = pd.read_csv('./comparison/B.csv')

        # Seleccionar las columnas de interés
        df_selected = df_2[['Training Loss', 'Validation Loss']]

        # Mostrar el gráfico de líneas
        st.line_chart(data=df_selected)
        
        st.markdown("Perdida")
        df_selected = df_2[['Training Accuracy','Validation Accuracy']]

        # Mostrar el gráfico de líneas
        st.line_chart(data=df_selected)
        
        # st.image(f"./comparison/{model_images[1]}", use_column_width=True, caption="mosquitos_v2")
        st.subheader("ACCURACY: 73.62%")
        st.write("Este modelo, al igual que el modelo anterior, es una red neuronal convolucional (CNN) que procesa imágenes de mosquitos u objetos similares. Comienza con tres capas de convolución, seguidas de capas de Batch Normalization y Max Pooling. La adición de Batch Normalization tiene como objetivo normalizar las activaciones intermedias de las capas convolucionales, lo que puede ayudar a acelerar el entrenamiento y a mejorar la convergencia del modelo. Luego, las capas completamente conectadas y la capa de dropout se utilizan para la clasificación, y el modelo se compila con el optimizador 'Adam' y la función de pérdida 'categorical_crossentropy'. En resumen, este modelo es similar al anterior en términos de su estructura, pero con la adición de Batch Normalization para mejorar el rendimiento y la convergencia del modelo durante el entrenamiento.")
    if model_to_use == "3 (80%)":
        
        st.markdown("Precisión")
        df_3 = pd.read_csv('./comparison/C.csv')

        # Seleccionar las columnas de interés
        df_selected = df_3[['Training Loss', 'Validation Loss']]
        
        # Mostrar el gráfico de líneas
        st.line_chart(data=df_selected)
        
        st.markdown("Perdida")

        df_selected = df_3[['Training Accuracy','Validation Accuracy']]

        # Mostrar el gráfico de líneas
        st.line_chart(data=df_selected)
        # st.image(f"./comparison/{model_images[2]}", use_column_width=True, caption="mosquitos_v2")
        st.subheader("ACCURACY:  80.64%")
        st.write("A diferencia de los modelos anteriores, este modelo opera con imágenes de mayor resolución (128x128 píxeles). Su estructura comprende tres capas de convolución para la extracción de características, seguidas de capas de Max Pooling para reducir la dimensionalidad de las características extraídas. Después, la salida se aplana y se conecta a dos capas completamente conectadas para la clasificación, con una capa de dropout para prevenir el sobreajuste. El modelo se compila con el optimizador 'Adam' y la función de pérdida 'categorical_crossentropy'. En resumen, este modelo se ajusta a imágenes más detalladas y sigue el enfoque típico de procesamiento de imágenes mediante convoluciones y capas completamente conectadas para la clasificación proporcionando en mayor accuracy posible.")
    # Show performance metrics and interactive graphs


elif selection == "Información extra":
    st.title("Información extra")
    
    
    st.subheader("Sabías que...")
    st.write("""Los mosquitos, esos pequeños insectos zumbadores que a menudo nos atormentan durante las noches de verano, tienen una serie de datos curiosos que los hacen fascinantes y, a la vez, peligrosos. Estas criaturas, principalmente las hembras, son responsables de la transmisión de diversas enfermedades, incluyendo el chikungunya, el dengue y la malaria, debido a su papel como vectores.

En primer lugar, los mosquitos son portadores de estos virus y parásitos mortales. Cuando una hembra pica a un humano infectado, ingiere el patógeno y luego lo transmite a su próxima víctima a través de su saliva mientras se alimenta de su sangre. Esto hace que los mosquitos sean eficientes agentes de propagación de enfermedades.

Además, los mosquitos son responsables de más muertes en todo el mundo que cualquier otro animal, debido a las enfermedades que transmiten. Aproximadamente 700 millones de personas contraen enfermedades transmitidas por mosquitos cada año, lo que ilustra la magnitud de su impacto en la salud pública.

Estos datos subrayan la importancia de tomar medidas preventivas, como el uso de repelentes y la eliminación de criaderos de mosquitos, para reducir el riesgo de enfermedades transmitidas por estos pequeños pero peligrosos insectos.""")
    st.subheader("Resultados")
    st.write(""" en los resultados y el análisis de un modelo de red neuronal desarrollado para identificar diversas especies de mosquitos a partir de imágenes. Se utilizó una base de datos con 10,700 imágenes reales de mosquitos, divididas en un 80% para el entrenamiento y un 20% para las pruebas. Las categorías de mosquitos incluyen tanto especies como géneros, con la particularidad de que la clasificación de las especies de Culex representa un desafío y, por lo tanto, se agrupan bajo un género. Además, se aborda la limpieza de datos, la configuración de parámetros y una comparación de los algoritmos utilizados en el modelo. Se destaca la creación de una aplicación con Streamlit que permite realizar predicciones de especies de mosquitos a partir de imágenes, mostrar información sobre los modelos y proporcionar detalles generales sobre el proyecto.""")

