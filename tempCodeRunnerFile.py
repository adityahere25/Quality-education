import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Function to load and preprocess data
def load_and_preprocess_data():
    # Get the current script directory
    script_directory = os.path.dirname(os.path.realpath(__file__))

    # Combine script directory with the filename
    file_path = "C:/Users/haris/OneDrive/Desktop/Python Program/student_data.csv"

    # Load data
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

        # Normalize or standardize features
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(data_filled), columns=data_filled.columns)

        # Feature Engineering
        data_scaled['attendance_rate'] = data_scaled['attendance'] / data_scaled['total_classes']
        data_encoded = pd.get_dummies(data_scaled, columns=['categorical_column'])

        return data_encoded

    else:
        st.error(f"File '{file_path}' not found. Please check the file path.")
        return None

# Streamlit App
st.title("Student Dropout Prediction App")

# Load and preprocess data
data = load_and_preprocess_data()

if data is not None:
    # Model Building
    model = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(data.drop("dropout", axis=1), data["dropout"], test_size=0.2)
    model.fit(X_train, y_train)

    # Model Evaluation
    predictions = model.predict(X_test)
    metrics_dict = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1-score": f1_score(y_test, predictions)
    }

    # Display evaluation metrics
    st.subheader("Model Evaluation Metrics")
    for metric, value in metrics_dict.items():
        st.write(f"{metric}:", value)

    # Prediction and Alert Generation for new student data
    st.subheader("Predict Dropout Risk for New Student")
    new_student_data = st.text_area("Enter new student data (CSV format):", "")
    if new_student_data:
        # Convert input to DataFrame
        new_student_data = pd.read_csv(pd.compat.StringIO(new_student_data))

        # Preprocess new data
        new_student_data_scaled = scaler.transform(imputer.transform(pd.get_dummies(new_student_data, columns=['categorical_column'])))
        
        # Predict dropout risk
        dropout_risk = model.predict_proba(new_student_data_scaled)[:, 1]

        # Display prediction
        st.write("Predicted Dropout Risk Probability:", dropout_risk)

        # Set a threshold for high-risk predictions and generate alert
        risk_threshold = 0.5
        if dropout_risk > risk_threshold:
         st.warning("High dropout risk! Generate alerts or take necessary actions.")
