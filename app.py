import streamlit as st
import joblib
import pandas as pd
import datetime as dt

# Load the pre-trained SARIMA model
model = joblib.load('gold_price_model.pkl')

# Title of the app
st.title("Gold Price Prediction App")

# Subtitle
st.subheader("Predict 24K Gold Price for a Future Date")

# User Input: Date
future_date = st.date_input(
    "Select a future date",
    min_value=dt.date.today(),
    help="Choose a date for which you want the predicted gold price."
)

# Display the selected date for debugging
st.write(f"Selected future date: {future_date}")

# Predict button
if st.button("Predict Price"):
    try:
        # Convert the future_date to pandas Timestamp
        future_date_ts = pd.Timestamp(future_date)

        # You need the last date in the model's trained data to make future predictions.
        # Example of the historical data
        historical_data = pd.DataFrame({
            '24k': [7400, 7420, 7450, 7480, 7510],  # Example recent gold prices
            'date': pd.date_range(start="2023-12-01", periods=5, freq="D")
        })
        historical_data.set_index('date', inplace=True)

        # Calculate forecast steps
        forecast_steps = (future_date_ts - historical_data.index[-1]).days  # Get the difference in days
        
        # Ensure the forecast is for a future date
        if forecast_steps < 1:
            st.error("Please select a future date that's later than the most recent date.")
        else:
            # Get the prediction using the model
            # We use predict() by providing start and end for the forecast range
            predicted_price = model.predict(start=historical_data.index[-1], end=future_date_ts)

            # Display the result
            st.success(f"The predicted price of 24K gold on {future_date} is ₹{predicted_price[-1]:.2f}")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
