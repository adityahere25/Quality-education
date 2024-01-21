import streamlit as st
import pandas as pd
import numpy as np

# Generate random student data
np.random.seed(10)  # For reproducibility

students = 100  # Number of students
features = ["age", "grades", "attendance", "family_income", "extracurricular_activities"]

data = pd.DataFrame({
    "age": np.random.randint(15, 25, size=students),
    "grades": np.random.randint(60, 100, size=students),
    "attendance": np.random.randint(70, 100, size=students),
    "family_income": np.random.randint(20000, 100000, size=students),
    "extracurricular_activities": np.random.randint(0, 5, size=students)
})

# Simulate dropouts with a random probability
dropout_prob = 0.1
data["dropout"] = np.random.binomial(1, dropout_prob, size=len(data))  # Adjusted size

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Display the first few rows of the data using Streamlit
st.title("Random Student Data")
st.subheader("First Few Rows:")
st.dataframe(data.head())

# Summary statistics using Streamlit
st.subheader("Summary Statistics:")
st.write(data.describe())

# Visualize data using Streamlit
st.subheader("Visualizing Data:")
st.line_chart(data[["age", "grades"]])

# Additional analysis or visualizations can be added as needed

# Note: To run this script in Streamlit, save it as a .py file and use the command `streamlit run filename.py`
