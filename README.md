# Casting-Defect-Detection

# Casting Defect Detection

This repository contains a machine learning project aimed at detecting defects in casting components using Convolutional Neural Networks (CNN). The solution is designed to automate the inspection process, improve accuracy, and reduce human error in defect identification.

## Features
- **Dataset**: Includes casting images with both defective and non-defective labels.
- **Model Architecture**: Utilizes a CNN model for feature extraction and classification.
- **Deployment**: Deployed as a web application using Streamlit for user interaction.
- **Evaluation Metrics**: Accuracy, precision, recall, and F1 score for performance measurement.

## Repository Structure
```
Casting-Defect-Detection/
├── data/                   # Dataset folder
├── models/                 # Saved trained models
├── notebooks/              # Jupyter notebooks for experimentation
├── app/                    # Streamlit app files
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── main.py                 # Main script for the Streamlit app
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tamilselvan-sde/Casting-Defect-Detection-
   cd Casting-Defect-Detection-
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Dataset
The dataset consists of casting images categorized into:
- **Defective**: Images with defects.
- **Non-defective**: Images without defects.

You can preprocess the dataset using the provided scripts in the `notebooks/` folder.

## Model Training
The model is trained using a CNN architecture. Training steps:
1. Load and preprocess the dataset.
2. Train the model using the `train_model.py` script.
3. Save the trained model for deployment.

## Deployment
The Streamlit app provides an intuitive interface for:
- Uploading casting images.
- Detecting whether the uploaded image is defective or non-defective.

Access the deployed app [here](https://tamilselvan-sde.streamlit.app/).

## Results
- **Accuracy**: Achieved over 95% accuracy on the test set.
- **Precision and Recall**: Ensures robust defect detection with minimal false positives.

## Future Scope
- Expand dataset with more diverse defect types.
- Optimize the CNN architecture for faster inference.
- Integrate real-time camera feed for live defect detection.

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request for any enhancements or fixes.

## License
This project is licensed under the MIT License.

## Author
[Tamilselvan S](https://github.com/tamilselvan-sde)

Feel free to reach out for questions or collaboration opportunities!
