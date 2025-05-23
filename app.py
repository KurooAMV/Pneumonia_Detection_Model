import os
import streamlit as st
import keras
import numpy as np
from keras.models import load_model
from PIL import Image

def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((128, 128))
    img = np.array(img)
    img = np.expand_dims(img, axis=-1)  # Add channel dimension: (128, 128, 1)
    img = np.expand_dims(img, axis=0)   # Add batch dimension: (1, 128, 128, 1)
    img = img / 255.0
    return img

model = keras.models.load_model("model/PneumoniaDetectionModel.keras")

st.title("Pneumonia Detector Using CNN")
uploaded_file = st.file_uploader("Upload an Image for Prediction", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    thumbnail = image_pil.copy()
    thumbnail.thumbnail((200, 200)) 
    st.image(thumbnail, caption="Preview", width=100)

    if st.button("Predict"):
        img_array = preprocess_image(image_pil)
        prediction = model.predict(img_array)
        predicted_class = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

        st.write("### Predicted Class:")
        if predicted_class == "Pneumonia":
            st.warning("Pneumonia")
        else:
            st.success("Normal")
        st.write(f"### Confidence Level: {confidence:.2%}")
