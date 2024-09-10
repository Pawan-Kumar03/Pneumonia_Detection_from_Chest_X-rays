import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the pre-trained model
model = load_model('modelvgg16.h5')

# Streamlit web app
st.title("Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to detect if it has pneumonia.")

# File uploader for image
uploaded_file = st.file_uploader("Choose an X-ray image...", type="jpg")

if uploaded_file is not None:
    # Open the image file
    img = Image.open(uploaded_file)

    # Convert image to RGB if it's not already
    if img.mode != 'RGB':
        img = img.convert('RGB')

    st.image(img, caption="Uploaded X-ray Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((256, 256))  # Resize as required by the model
    img_array = np.array(img) / 255.0

    # Debug: Print the shape of the array
    st.write("Image array shape:", img_array.shape)

    # Ensure the array has 3 channels
    if img_array.shape[-1] == 3:
        img_array = img_array.reshape(1, 256, 256, 3)

        # Predict using the model
        prediction = model.predict(img_array)

        # Show results
        if prediction[0] > 0.5:
            st.write("Prediction: **Pneumonia**")
        else:
            st.write("Prediction: **Normal**")
    else:
        st.write("Error: The image does not have 3 channels.")
