import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
# from tensorflow.keras.models import load_model

# Define the class labels
class_labels = {0: 'Healthy', 1: 'Type 1 disease', 2: 'Type 2 disease'}

# Load the trained model
cnn = tf.keras.models.load_model('model.h5')

# Function to make predictions
@tf.function
def predict(image):
    result = cnn(image)  # Call the model directly on the input tensor
    label_index = tf.argmax(result, axis=1)  # Find the index of the maximum value
    return label_index

# Streamlit app
def main():
    st.markdown("<h1 style='color: yellow; font-family: Arial, sans-serif; font-size: 24px;'>Lung disease Detector App by CodeWudaya 👌</h1>", unsafe_allow_html=True)
    st.write("Upload an image and the app will predict its class :🫁")

    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')  # Open the image using PIL
        image = image.resize((64, 64))  # Resize the image to the desired size
        image = np.array(image)  # Convert the image to a NumPy array
        image = np.expand_dims(image, axis=0)  # Add an extra dimension

        result = predict(image)
        prediction_index = result.numpy()[0]
        prediction_label = class_labels[prediction_index]

        
        if prediction_label == 'Healthy':
            prediction_color = '#00FF00'  # Green color
        else:
            prediction_color = '#FF0000'  # Red color
        st.markdown(f"<p style='font-size:24px;font-weight:bold;color:{prediction_color};'>{prediction_label}</p>", unsafe_allow_html=True)
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.balloons()
if __name__ == "__main__":
    main()