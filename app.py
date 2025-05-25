import streamlit as st
import numpy as np
import keras
from keras.models import load_model
from PIL import Image
import os

def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((64, 64))  # Resize to match model input
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=-1)  # Add channel dimension: (64, 64, 1)
    img = np.expand_dims(img, axis=0)   # Add batch dimension: (1, 64, 64, 1)
    return img

def build_model():
    dimen = 64
    channels = 1
    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape = (dimen,dimen, channels)),
        keras.layers.InputLayer(input_shape = (dimen,dimen, channels)),
        keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(activation='relu',units=128),
        keras.layers.Dense(activation='sigmoid',units=1),
    ])
    return model

model = build_model()
model.load_weights("weights.h5")

# model = load_model("model/Pneumonia_Detector.keras", compile = False)
# weights = model.get_weights()
# model.set_weights(weights)
# model.load_weights("weights.h5")


st.title("Pneumonia Detector Using CNN")
col1, col2 = st.columns(2)

choice = st.radio(
    "Choose how to provide an image:",
    ["Upload your own", "Use sample images"],
    horizontal=True
)

image = None

if choice == "Upload your own":
    uploaded_file = col1.file_uploader("Upload an X-ray image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
elif choice == "Use sample images":
    sample_path = "static/samples"
    sample_images = [img for img in os.listdir(sample_path) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
    selected_sample = col1.selectbox("Select a sample image", sample_images)
    if selected_sample:
        image_path = os.path.join(sample_path, selected_sample)
        image = Image.open(image_path)

if image:
    image_pil = Image.open(image)
    col1.image(image, caption=f"Preview", use_container_width=True)

    if col1.button("Predict"):
        img_array = preprocess_image(image_pil)
        prediction = model.predict(img_array)
        predicted_class = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

        col2.write("### Predicted Class:")
        if predicted_class == "Pneumonia":
            col2.warning("Pneumonia")
        else:
            col2.success("Normal")
        col2.write(f"### Confidence Level: {confidence:.2%}")
