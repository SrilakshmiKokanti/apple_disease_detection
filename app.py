import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('apple_disease_model.h5')
    return model

model = load_model()

# Class names
class_names = ['Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___Scab', 'Apple___healthy']

def predict_disease(img):
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    return predicted_class, confidence

st.title('Apple Leaf Disease Detection')
st.write('Upload an image of an apple leaf to predict the disease.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_container_width=True)
    st.write("Predicting...")

    predicted_class, confidence = predict_disease(img)

    st.write(f'Predicted class: {predicted_class}')
    st.write(f'Confidence: {confidence:.4f}')