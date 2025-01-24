import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

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

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Display each uploaded image
        st.image(uploaded_file, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

        # Preprocess the image
        img_size = (128, 128)  # Ensure it matches the training input size
        image = Image.open(uploaded_file)  # Open image with PIL
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
    st.write("No files uploaded. Please upload one or more images to begin classification.")

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align: center; color: #aaa;">Powered by TensorFlow and Streamlit | © 2025 Casting Defect Detection</p>
    """,
    unsafe_allow_html=True
)
