import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import glob

# Ensure set_page_config is the first Streamlit command
st.set_page_config(
    page_title="Casting Defect Detection",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the TensorFlow Lite model
@st.cache_resource
def load_tflite_model():
    tflite_model_path = 'cnn_model_quantized.tflite'
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Get input and output details from the TFLite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
class_labels = {0: 'Not Defective', 1: 'Defective'}

# App Title
st.markdown(
    """
    <style>
    .title {
        font-size: 2.5em;
        color: #4CAF50;
        text-align: center;
    }
    .result {
        font-size: 1.5em;
        color: #FF5722;
        text-align: center;
        margin-top: 20px;
    }
    .confidence {
        font-size: 1.2em;
        color: #009688;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">Casting Defect Detection 🔧</h1>', unsafe_allow_html=True)
st.write("Upload multiple images to classify them as **Defective** or **Not Defective**.")

# File uploader to allow multiple uploads
uploaded_files = st.file_uploader(
    "Upload images (JPEG/PNG):", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True,
    help="Upload casting images to detect defects."
)

# List files in the current directory
current_dir = os.getcwd()  # Get the current directory (same as app.py)
jpeg_files = glob.glob(os.path.join(current_dir, "*.jpeg"))  # Search for .jpeg files

# Process selected or uploaded files
selected_file = None
if not uploaded_files and jpeg_files:
    st.write("No files uploaded. Select a test image from the current directory:")
    selected_file = st.selectbox("Available Test Images", jpeg_files)  # Allow user to select a file

if uploaded_files or selected_file:
    files_to_process = uploaded_files if uploaded_files else [selected_file]

    for file in files_to_process:
        # Handle files: uploaded file or selected from directory
        if uploaded_files:
            image = Image.open(file)  # Open directly from uploaded files
        else:
            image = Image.open(file)  # Open the selected file path

        st.image(image, caption=f"Test Image: {os.path.basename(file)}", use_column_width=True)

        # Preprocess the image
        img_size = (128, 128)  # Ensure it matches the training input size
        image = image.resize(img_size)  # Resize to match model input
        img_array = img_to_array(image) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Add batch dimension and convert to float32

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(prediction)  # Get class with highest probability

        # Display the result with confidence
        result_label = class_labels[predicted_class]
        confidence = prediction[0][predicted_class] * 100  # Confidence score

        st.markdown('<h2 class="result">Prediction: {}</h2>'.format(result_label), unsafe_allow_html=True)
        st.markdown('<p class="confidence">Confidence: {:.2f}%</p>'.format(confidence), unsafe_allow_html=True)
        st.markdown("---")  # Separator for each image

else:
    st.write("No images available to process.")

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align: center; color: #aaa;">Powered by TensorFlow and Streamlit | © 2025 Casting Defect Detection</p>
    """,
    unsafe_allow_html=True
)
