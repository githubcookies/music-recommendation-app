import joblib
from train_model import extract_features
import numpy as np
import os
import logging
import soundfile as sf
import librosa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_audio_file(file_path):
    """Verify if the audio file is valid and can be processed."""
    try:
        # Check basic file properties
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.error("File is empty")
            return False
            
        # Try reading with soundfile
        try:
            with sf.SoundFile(file_path) as audio_file:
                logger.info(f"Audio file details: {audio_file.samplerate}Hz, {audio_file.channels} channels, {audio_file.frames} frames")
                if audio_file.frames == 0:
                    logger.error("Audio file has no frames")
                    return False
        except Exception as e:
            logger.warning(f"SoundFile couldn't read the file: {str(e)}")
            # Don't return False here, try librosa as backup
            
        # Try reading with librosa
        try:
            y, sr = librosa.load(file_path, duration=1, sr=None)  # Just try loading 1 second
            logger.info(f"Successfully loaded audio with librosa: {sr}Hz sample rate")
            return True
        except Exception as e:
            logger.error(f"Librosa couldn't read the file: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error verifying audio file: {str(e)}")
        return False

def predict_healing_music(audio_path):
    """
    Predict whether a music file is healing or not.
    Returns: probability of being healing music (0-1)
    """
    try:
        # Step 1: Verify audio file
        logger.info(f"Starting prediction for file: {audio_path}")
        if not verify_audio_file(audio_path):
            logger.error("Audio file verification failed")
            return None
            
        # Step 2: Load model and scaler
        try:
            if not os.path.exists('model.joblib') or not os.path.exists('scaler.joblib'):
                logger.error("Model or scaler file not found")
                return None
                
            model = joblib.load('model.joblib')
            scaler = joblib.load('scaler.joblib')
            logger.info("Model and scaler loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model or scaler: {str(e)}")
            return None
        
        # Step 3: Extract features
        try:
            features = extract_features(audio_path)
            if features is None:
                logger.error("Feature extraction failed")
                return None
                
            # Verify feature dimensions
            expected_features = 33  # Based on our feature extraction
            if len(features) != expected_features:
                logger.error(f"Incorrect number of features. Expected {expected_features}, got {len(features)}")
                return None
                
            logger.info(f"Successfully extracted {len(features)} features")
        except Exception as e:
            logger.error(f"Error during feature extraction: {str(e)}")
            return None
        
        # Step 4: Scale features
        try:
            features_scaled = scaler.transform(features.reshape(1, -1))
            logger.info("Features scaled successfully")
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            return None
        
        # Step 5: Make prediction
        try:
            probability = model.predict_proba(features_scaled)[0][1]
            logger.info(f"Prediction successful: {probability:.2f}")
            return probability
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        return None
