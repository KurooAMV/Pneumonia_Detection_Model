import os
import streamlit as st
import tensorflow as tf
import keras
import numpy as np
from keras.models import load_model
from PIL import Image
import gdown

def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((128, 128))
    img = np.array(img)
    img = np.expand_dims(img, axis=-1)  # Add channel dimension: (128, 128, 1)
    img = np.expand_dims(img, axis=0)   # Add batch dimension: (1, 128, 128, 1)
    img = img / 255.0
    return img

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
uploaded_file = st.file_uploader("Upload an Image for Prediction", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    # st.image(image_pil, caption="Uploaded X-ray", use_column_width=True)
    thumbnail = image_pil.copy()
    thumbnail.thumbnail((200, 200)) 
    st.image(thumbnail, caption="Preview", width=100)

    if st.button("Predict"):
        # download_model()
        # model = load_model()
        download_model()
        model = keras.models.load_model(MODEL_PATH)
        img_array = preprocess_image(image_pil)
        prediction = model.predict(img_array)
        os.remove(MODEL_PATH)
        predicted_class = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

        st.write("### Predicted Class:")
        if predicted_class == "Pneumonia":
            st.warning("Pneumonia")
        else:
            st.success("Normal")
        st.write(f"### Confidence Level: {confidence:.2%}")
