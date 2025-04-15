import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the model
model = tf.keras.models.load_model('apple_disease_model.h5')

# Class names
class_names = ['Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___Scab', 'Apple___healthy']

def predict_disease(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    return predicted_class, confidence

if __name__ == "__main__":
    # Example usage
    image_path = 'data/val/Apple___Black_rot/image (1000).JPG'  # Replace with your image path
    predicted_class, confidence = predict_disease(image_path)
    print(f'Predicted class: {predicted_class}')
    print(f'Confidence: {confidence:.4f}')