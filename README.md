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
├── app.py                   # Streamlit app files
├── cast_ok_0_942.jpeg       # Sample test image
├── vibration_damper_failure_detection.ipynb  # Jupyter notebook for experimentation
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
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

## Vibration Damper Failure Detection Documentation

### Introduction
This document serves as a comprehensive guide to the vibration damper failure detection process. It outlines the methodology, tools, and results obtained during the analysis of vibration dampers to identify potential failures.

### Objective
The primary objective of this analysis is to:
- Detect failures in vibration dampers.
- Provide insights into failure patterns.
- Suggest corrective measures to enhance damper reliability.

### Tools and Technologies Used
- **Programming Language**: Python
- **Libraries**: NumPy, pandas, matplotlib, seaborn, SciPy
- **Machine Learning Frameworks**: scikit-learn
- **Notebook Environment**: Jupyter Notebook, Colab

### Data Overview
- **Data Source**: Sensor data from vibration dampers.
- **Data Structure**: Includes time-series data, frequency domain information, and environmental conditions.

### Methodology
#### Data Collection:
- Gathered real-time sensor data from operational dampers.
- Cleaned and preprocessed data to remove noise and outliers.

#### Exploratory Data Analysis (EDA):
- Analyzed data patterns and distributions.
- Visualized trends in vibration and environmental factors.

#### Feature Engineering:
- Extracted key features such as mean vibration amplitude and peak frequencies.
- Calculated derived metrics like stress variance and temperature correlation.

#### Failure Detection Model:
- Developed machine learning models (Random Forest, SVM) to predict failure likelihood.
- Evaluated models using metrics like accuracy, precision, recall, and F1-score.

#### Validation and Testing:
- Split data into training and testing sets (80-20 split).
- Cross-validated model performance.

### Results
#### Key Findings:
- High vibration amplitude and frequency spikes correlate with failures.
- Extreme operating temperatures exacerbate damper degradation.

#### Model Performance:
- Random Forest achieved 92% accuracy with a 0.89 F1-score.
- SVM model demonstrated 88% accuracy but required more computational resources.

### Recommendations
- Regular monitoring of vibration amplitude and temperature.
- Implement predictive maintenance strategies using the developed models.
- Optimize operating conditions to minimize stress on dampers.

### Conclusion
The vibration damper failure detection framework developed provides a robust mechanism for identifying potential failures early. By leveraging advanced data analysis and machine learning techniques, this solution enhances the reliability and longevity of vibration dampers.

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request for any enhancements or fixes.

## License
This project is licensed under the MIT License.

## Author
[Tamilselvan S](https://github.com/tamilselvan-sde)

Feel free to reach out for questions or collaboration opportunities!
