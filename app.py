from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Load the trained Random Forest model, scaler, and label encoder
with open('random_forest_weather_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.json
    
    # Extract features from JSON
    try:
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        wind_speed = float(data['windSpeed'])
        pressure = float(data['pressure'])
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'Invalid input data'}), 400
    
    # Create feature array for prediction
    features = np.array([[temperature, humidity, wind_speed, pressure]])
    
    # Scale the input features using the loaded scaler
    features_scaled = scaler.transform(features)
    
    # Make prediction using the model
    prediction = model.predict(features_scaled)
    
    # Decode the predicted class back to original label
    weather_condition = label_encoder.inverse_transform(prediction)[0]
    
    # Return prediction as JSON
    return jsonify({'prediction': weather_condition})

if __name__ == '__main__':
    app.run(debug=True)