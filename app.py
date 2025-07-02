import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('dog_cat_model.h5')

# Streamlit app title
st.title("🐶🐱 Dog vs Cat Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for prediction
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make a prediction
    prediction = model.predict(img_array)[0][0]
    label = "🐱 Cat" if prediction < 0.5 else "🐶 Dog"
    confidence = (1 - prediction) * 100 if prediction < 0.5 else prediction * 100

    # Show the prediction result
    st.write(f"**Prediction:** {label} ({confidence:.2f}% confidence)")
