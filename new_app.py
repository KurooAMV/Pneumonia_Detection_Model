import os
import streamlit as st
import tensorflow as tf
import keras
import numpy as np
from keras.models import load_model
from PIL import Image
import gdown

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Adjust target_size if needed
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

dimen = 64
MODEL_PATH = 'model/pneumonia_detection_model.keras'
GDRIVE_ID = '1gHoNs4ulIA8HAd29f3uRSMTOTXLcV2MQ1GpHz4Q2JgsdBWGZMNANX4MAgCUtxmUIE'
GDRIVE_URL = f'https://drive.google.com/uc?id={GDRIVE_ID}'

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        # print("Downloading model from Google Drive...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    return

st.title("Pneumonia Detector Using CNN")
uploaded_file = st.file_uploader("Upload an Image for Prediction", type = ['jpg','png','jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("Predict"):
        download_model()
        model = keras.models.load_model(MODEL_PATH)
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        predicted_class = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

        st.write("### Predicted Class:")
        if predicted_class == "Pneumonia":
            st.warning("Pneumonia")
        else:
            st.success("Normal")
        st.write(f"### Confidence Level: {confidence:.2%}")
