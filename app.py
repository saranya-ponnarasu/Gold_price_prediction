import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model (replace 'your_model.pkl' with the actual path to your .pkl file)
model = joblib.load('your_model.pkl')

# Set the title of the app
st.title('Machine Learning Model Deployment with Streamlit')

# Input fields for user data
st.header('Enter the required inputs for prediction')

# Example features
feature1 = st.number_input('Feature 1 (e.g., age)', min_value=0, max_value=100, value=25)
feature2 = st.number_input('Feature 2 (e.g., salary)', min_value=0, max_value=100000, value=50000)
feature3 = st.number_input('Feature 3 (e.g., hours worked)', min_value=0, max_value=168, value=40)

# Create a DataFrame from the input data
input_data = pd.DataFrame(np.array([[feature1, feature2, feature3]]), columns=['feature1', 'feature2', 'feature3'])

# Display the input data
st.write('Input data:')
st.write(input_data)

# Prediction button
if st.button('Predict'):
    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Show the prediction result
    st.write('Prediction Result:')
    st.write(prediction)
