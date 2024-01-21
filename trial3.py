import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to load and preprocess data
def load_and_preprocess_data():
    # Directly load data from a predefined CSV file
    file_path = "C:/Users/haris/OneDrive/Desktop/Python Program/student_data.csv"

    if os.path.exists(file_path):
        # Load data
        data = pd.read_csv(file_path)

        # Separate numeric and non-numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        non_numeric_columns = list(set(data.columns) - set(numeric_columns))

        # Handle missing values for numeric columns
        imputer = SimpleImputer(strategy='mean')
        data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

        # Drop rows with missing values in non-numeric columns
        data = data.dropna(subset=non_numeric_columns)

        # Preprocessing for categorical data
        categorical_columns = data.select_dtypes(include=['object']).columns
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Preprocessing for numeric data
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_columns),
                ('cat', categorical_transformer, categorical_columns)
            ])

        # Feature Engineering
        data['attendance_rate'] = data['Attendance'] / data['Total_classes']

        # Apply preprocessing
        data_encoded = pd.DataFrame(preprocessor.fit_transform(data), columns=numeric_columns.tolist() + preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_columns).tolist())

        return data_encoded

    else:
        st.error(f"File '{file_path}' not found. Please check the file path.")
        return None

# Streamlit App
def main():
    st.title("Student Dropout Prediction App")

    # Load and preprocess data
    data = load_and_preprocess_data()

    if data is not None:
        # Model Building
        model = RandomForestClassifier()
        X_train, X_test, y_train, y_test = train_test_split(data.drop(['Dropout_No', 'Dropout_Yes'], axis=1), data['Dropout_Yes'], test_size=0.2)


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
            new_student_data_encoded = pd.DataFrame(preprocessor.transform(new_student_data), columns=numeric_columns.tolist() + preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_columns).tolist())

            # Predict dropout risk
            dropout_risk = model.predict_proba(new_student_data_encoded)[:, 1]

            # Display prediction
            st.write("Predicted Dropout Risk Probability:", dropout_risk)

            # Set a threshold for high-risk predictions and generate alert
            risk_threshold = 0.5
            if dropout_risk > risk_threshold:
                st.warning("High dropout risk! Generate alerts or take necessary actions.")

if __name__ == "__main__":
    main()
