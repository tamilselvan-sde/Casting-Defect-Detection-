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
    .summary {
        font-size: 1.2em;
        color: #555;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">Casting Defect Detection 🔧</h1>', unsafe_allow_html=True)
st.markdown(
    """
    Welcome to the **Casting Defect Detection** tool. This app uses a TensorFlow Lite model to classify casting products 
    as either **Defective** or **Not Defective**. Choose one of the tabs below to upload an image, view default images, 
    access the model summary, explore tech stacks, view documentation, or visit the GitHub repository.
    """,
    unsafe_allow_html=True
)

# Create Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Upload Image", "Default Images", "Model Summary", "Tech Stacks Used", "Documentation", "GitHub"
])

# Upload Image Tab
with tab1:
    st.header("Upload an Image")
    st.write("Upload your casting product image for defect detection.")
    uploaded_file = st.file_uploader(
        "Upload a JPEG/PNG image:", 
        type=["jpg", "jpeg", "png"], 
        help="Upload a casting image to detect defects."
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

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

# Default Images Tab
with tab2:
    st.header("Select a Default Image")
    st.write("Choose a test image from the available default images for prediction.")
    current_dir = os.getcwd()  # Get the current directory (same as app.py)
    jpeg_files = glob.glob(os.path.join(current_dir, "*.jpeg"))  # Search for .jpeg files

    if jpeg_files:
        selected_file = st.selectbox("Available Default Images", jpeg_files)
        if selected_file:
            image = Image.open(selected_file)
            st.image(image, caption=f"Default Image: {os.path.basename(selected_file)}", use_column_width=True)

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
    else:
        st.write("No default images found in the current directory.")

# Model Summary Tab
with tab3:
    st.header("Model Summary")
    st.markdown(
        """
        This tab provides information about the TensorFlow Lite model used for predictions.
        Below is the input and output details of the loaded model:
        """,
        unsafe_allow_html=True
    )

    # Display Input Details
    st.subheader("Input Details")
    for input_detail in input_details:
        st.markdown(
            f"""
            - **Name**: {input_detail['name']}
            - **Shape**: {input_detail['shape']}
            - **Data Type**: {input_detail['dtype']}
            """,
            unsafe_allow_html=True
        )

    # Display Output Details
    st.subheader("Output Details")
    for output_detail in output_details:
        st.markdown(
            f"""
            - **Name**: {output_detail['name']}
            - **Shape**: {output_detail['shape']}
            - **Data Type**: {output_detail['dtype']}
            """,
            unsafe_allow_html=True
        )

# Tech Stacks Tab
with tab4:
    st.header("Tech Stacks Used")
    st.write("This project is built using the following technologies:")

    tech_stacks = [
        {"name": "TensorFlow", "logo": "https://www.tensorflow.org/images/tf_logo_social.png"},
        {"name": "Keras", "logo": "https://keras.io/img/logo-small.png"},
        {"name": "NumPy", "logo": "https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg"},
        {"name": "Scikit-learn", "logo": "https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg"},
        {"name": "Pandas", "logo": "https://pandas.pydata.org/static/img/pandas_white.svg"},
        {"name": "Matplotlib", "logo": "https://matplotlib.org/stable/_static/logo2_compressed.svg"},
        {"name": "Streamlit", "logo": "https://streamlit.io/images/brand/streamlit-logo-primary-colormark-lighttext.png"},
        {"name": "GitHub", "logo": "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"},
        {"name": "MLflow", "logo": "https://mlflow.org/docs/latest/_static/MLflow-logo-final-black.png"},
        {"name": "Kafka", "logo": "https://upload.wikimedia.org/wikipedia/commons/0/0a/Apache_kafka-icon.svg"},
        {"name": "Docker", "logo": "https://www.docker.com/wp-content/uploads/2022/03/Moby-logo.png"}
    ]

    cols = st.columns(3)  # Create a 3-column layout
    for i, tech in enumerate(tech_stacks):
        with cols[i % 3]:  # Distribute logos across the columns
            st.image(tech["logo"], width=100)
            st.markdown(f"**{tech['name']}**")

# Documentation Tab
with tab5:
    st.header("Documentation")
    st.markdown(
        """
        Access the full documentation for this project by clicking the link below:
        - [Project Documentation](https://docs.google.com/document/d/1LpcYPT0w4vP473ERT9twR7Jtwt7H7WgCufTlPOhAYUI/edit?usp=sharing)
        """,
        unsafe_allow_html=True
    )

# GitHub Tab
with tab6:
    st.header("GitHub Repository")
    st.markdown(
        """
        View the source code and contribute to this project by visiting the GitHub repository:
        - [GitHub Repository](https://github.com/tamilselvan-sde/Casting-Defect-Detection-/tree/main)
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align: center; color: #aaa;">Powered by TensorFlow and Streamlit | © 2025 Casting Defect Detection</p>
    """,
    unsafe_allow_html=True
)
