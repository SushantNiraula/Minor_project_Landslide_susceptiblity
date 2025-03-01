from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('landslide_model_tuned.pkl')
scaler = joblib.load('scaler.pkl')

# Store predictions for history
if not os.path.exists('predictions'):
    os.makedirs('predictions')

# Risk levels and their descriptions
risk_descriptions = {
    'Low': 'No immediate risk of landslide.',
    'Moderate': 'Some risk factors present, monitoring advised.',
    'High': 'Significant risk factors detected, prepare for possible evacuation.',
    'Very High': 'Extreme danger, immediate evacuation recommended!'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from ESP32
        data = request.json
        
        # Extract features
        features = [
            data.get('temperature', 0),
            data.get('humidity', 0),
            data.get('precipitation', 0),
            data.get('soil_moisture', 0),
            data.get('elevation', 0)
        ]
        
        # Convert to DataFrame
        input_df = pd.DataFrame([features], columns=[
            'Temperature (°C)', 
            'Humidity (%)', 
            'Precipitation (mm)', 
            'Soil Moisture (%)', 
            'Elevation (m)'
        ])
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Get class indices and their probabilities
        classes = model.classes_
        probabilities = {cls: float(prob) for cls, prob in zip(classes, prediction_proba)}
        
        # Create response
        response = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'prediction': prediction,
            'description': risk_descriptions.get(prediction, "Unknown risk level"),
            'probabilities': probabilities,
            'raw_data': {
                'temperature': features[0],
                'humidity': features[1],
                'precipitation': features[2],
                'soil_moisture': features[3],
                'elevation': features[4]
            }
        }
        
        # Save prediction to file
        with open(f'predictions/prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(response, f)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
