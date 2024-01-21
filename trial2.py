import streamlit as st
import pandas as pd
import numpy as np

# Generate random student data (same as before)
np.random.seed(10)
students = 100
features = ["age", "grades", "attendance", "family_income", "extracurricular_activities"]
data = pd.DataFrame({
    "age": np.random.randint(15, 25, size=students),
    "grades": np.random.randint(60, 100, size=students),
    "attendance": np.random.randint(70, 100, size=students),
    "family_income": np.random.randint(20000, 100000, size=students),
    "extracurricular_activities": np.random.randint(0, 5, size=students)
})
data["dropout"] = np.random.binomial(1, 0.1, size=students)
data = data.sample(frac=1)

# Handle missing values using removal
st.title("Handling Missing Values")
st.subheader("Original Data:")
st.dataframe(data)

# Check for missing values initially
st.subheader("Initial Missing Values Check:")
st.write(data.isnull().sum())

threshold = 0.2  # Set threshold for acceptable missingness

# Drop columns with excessive missing values
data = data.dropna(axis=1, thresh=int(threshold * len(data)))

# Drop rows with missing values in specified columns
data = data.dropna(subset=features)

# Display the cleaned data
st.subheader("Cleaned Data:")
st.dataframe(data)

# Verify removal
st.subheader("Final Missing Values Check:")
st.write(data.isnull().sum())

# Check the remaining dataset size
st.subheader("Remaining Dataset Size:")
st.write(data.shape)

# Prepare for feature normalization or standardization
# ... (code for normalization/standardization techniques)
