import joblib
from train_model import extract_features
import numpy as np

def predict_healing_music(audio_path):
    """
    Predict whether a music file is healing or not.
    Returns: probability of being healing music (0-1)
    """
    try:
        # Load model and scaler
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        
        # Extract features
        features = extract_features(audio_path)
        if features is None:
            return None
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Predict
        probability = model.predict_proba(features_scaled)[0][1]
        return probability
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None
