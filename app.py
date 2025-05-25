import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((64, 64))  # Resize to match model input
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=-1)  # Add channel dimension: (64, 64, 1)
    img = np.expand_dims(img, axis=0)   # Add batch dimension: (1, 64, 64, 1)
    return img

model = load_model("model/Pneumonia_Detector.keras")
weights = model.get_weights()
model.set_weights(weights)
# model.load_weights("weights.h5")


st.title("Pneumonia Detector Using CNN")
uploaded_file = st.file_uploader("Upload an Image for Prediction", type=['jpg', 'png', 'jpeg'])
col1, col2 = st.columns(2)

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    thumbnail = image_pil.copy()
    thumbnail.thumbnail((200, 200)) 
    col1.image(thumbnail, caption="Preview", width=100)

    if col1.button("Predict"):
        img_array = preprocess_image(image_pil)
        prediction = model.predict(None,img_array)
        predicted_class = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

        col2.write("### Predicted Class:")
        if predicted_class == "Pneumonia":
            col2.warning("Pneumonia")
        else:
            col2.success("Normal")
        col2.write(f"### Confidence Level: {confidence:.2%}")
