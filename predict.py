import joblib
from train_model import extract_features
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_healing_music(audio_path):
    """
    Predict whether a music file is healing or not.
    Returns: probability of being healing music (0-1)
    """
    try:
        # Check if file exists and is readable
        if not os.path.exists(audio_path):
            logger.error(f"File not found: {audio_path}")
            return None
            
        # Check file size
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            logger.error("File is empty")
            return None
            
        logger.info(f"Processing file: {audio_path} (size: {file_size} bytes)")
        
        # Load model and scaler
        if not os.path.exists('model.joblib') or not os.path.exists('scaler.joblib'):
            logger.error("Model or scaler file not found")
            return None
            
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        logger.info("Model and scaler loaded successfully")
        
        # Extract features
        features = extract_features(audio_path)
        if features is None:
            logger.error("Feature extraction failed")
            return None
        
        # Check feature dimensions
        expected_features = 33  # Based on our feature extraction
        if len(features) != expected_features:
            logger.error(f"Incorrect number of features. Expected {expected_features}, got {len(features)}")
            return None
            
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Predict
        probability = model.predict_proba(features_scaled)[0][1]
        logger.info(f"Prediction successful: {probability}")
        return probability
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return None
