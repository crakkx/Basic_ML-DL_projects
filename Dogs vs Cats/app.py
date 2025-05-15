import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("models/catvsdog.h5")

st.title("🐶🐱 Dog vs Cat Classifier")

st.write("Upload an image, and the model will predict if it's a dog or a cat.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.resize((200,200))  
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image',use_container_width=True)

    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success(f"Prediction: 🐶 Dog ({prediction:.2f})")
    else:
        st.success(f"Prediction: 🐱 Cat ({1 - prediction:.2f})")
