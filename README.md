# Accident Severity Predictor

This is a Streamlit web app for predicting the severity of road accidents using a Random Forest Classifier with a pre-processing pipeline.

## Features
- Handles numeric and categorical features
- Encodes and scales data using a pipeline
- Predicts accident severity: Slight Injury, Serious Injury, Fatal Injury

## Dataset
- Contains road accident records with features like vehicle type, driver age, weather conditions, etc.
- **Note:** The target variable (`Accident_severity`) is imbalanced, which may affect model performance. Slight Injury occurs much more frequently than Serious or Fatal Injuries.

## How to Run
1. Clone the repository
2. Install dependencies:
   joblib
   numpy
   pandas
   streamlit
