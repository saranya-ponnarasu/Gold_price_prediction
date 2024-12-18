from flask import Flask
import joblib

app = Flask(__name__)

# Load your model (replace 'your_model.pkl' with your actual model file)
model = joblib.load('gold_price_model.pkl')

@app.route('/')
def home():
    return "Welcome to the Gold Price Prediction API!"

@app.route('/predict', methods=['GET'])
def predict():
    # Example prediction code, adjust to your specific model and input
    # If your model expects input via a GET request, pass the inputs accordingly
    prediction = model.predict([[1.5, 22.0]])  # Example input values, change as necessary
    return f"Predicted Price: {prediction[0]}"

# Handle favicon requests
@app.route('/favicon.ico')
def favicon():
    return '', 204  # Return a blank response for favicon requests

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)
