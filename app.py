import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import urllib.request

# 1. Page Title
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")
st.title("🧠 Brain Tumor Multi-Classification App")

# 2. Download Model From Google Drive Automatically
@st.cache_resource
def load_my_model():
    model_path = "tumor_model.keras"
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading your 145MB Inception model. Please wait a moment..."):
            google_drive_url = "https://drive.google.com/file/d/1LfuIpjHe3SXjFPArQcoroqH3qd3kef91/view?usp=sharing"
            
            # This line converts regular link into a direct download link
            if "file/d/" in google_drive_url:
                file_id = google_drive_url.split("/d/")[1].split("/")[0]
                direct_download_url = f"https://google.com{file_id}"
            else:
                direct_download_url = google_drive_url
                
            urllib.request.urlretrieve(direct_download_url, model_path)
            
    # Clear memory background processes to prevent crashes
    tf.keras.backend.clear_session()
    return tf.keras.models.load_model(model_path)

# Load the model
try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error loading model. Make sure your link is public. Details: {e}")
    st.stop()

# 3. Web Interface for User
uploaded_file = st.file_uploader("Upload an MRI Scan (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the image to the user
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
    
    if st.button("Run Diagnostic Prediction"):
        with st.spinner("Analyzing image..."):
            # Prepare image for Inception model (299x299 pixels)
            resized_image = image.resize((299, 299))
            img_array = tf.keras.preprocessing.image.img_to_array(resized_image)
            img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array)
            classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
            highest_idx = np.argmax(predictions)
            
            # Display Final Result
            st.subheader(f"Prediction: {classes[highest_idx]}")
            st.write(f"Confidence: {predictions[0][highest_idx]*100:.2f}%")
