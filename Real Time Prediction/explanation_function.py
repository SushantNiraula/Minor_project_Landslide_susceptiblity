import numpy as np

def explain_prediction(features, prediction, probabilities):
    """Generate an explanation for landslide risk prediction."""
    # Code from the explain_prediction function
    feature_names = ['Temperature', 'Humidity', 'Precipitation', 'Soil Moisture', 'Elevation']
    top_factors = []

    # Your model's feature importances
    # Sort features by importance
    sorted_idx = np.argsort(importances)[::-1]

    # Get the top 3 important features
    for i in sorted_idx[:3]:
        feature_name = feature_names[i]
        value = features[i]

        # Check if feature value is high or low relative to averages
        if feature_name == 'Humidity' and value > 80:
            top_factors.append(f"High humidity ({value}%)")
        elif feature_name == 'Precipitation' and value > 150:
            top_factors.append(f"High precipitation ({value}mm)")
        elif feature_name == 'Soil Moisture' and value > 80:
            top_factors.append(f"High soil moisture ({value}%)")
        elif feature_name == 'Elevation' and value > 700:
            top_factors.append(f"High elevation ({value}m)")

    # Create explanation based on prediction
    risk_level = prediction

    if risk_level == 'Low':
        explanation = f"The landslide risk is LOW. Main factors: {', '.join(top_factors)}"
    elif risk_level == 'Moderate':
        explanation = f"The landslide risk is MODERATE. Please monitor conditions. Main factors: {', '.join(top_factors)}"
    elif risk_level == 'High':
        explanation = f"WARNING: The landslide risk is HIGH. Prepare for possible evacuation. Main factors: {', '.join(top_factors)}"
    else:  # Very High
        explanation = f"DANGER: The landslide risk is VERY HIGH. Immediate evacuation recommended! Main factors: {', '.join(top_factors)}"

    # Add probability information
    max_prob = max(probabilities)
    explanation += f" (Confidence: {max_prob:.1%})"

    return explanation
