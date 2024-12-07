import joblib
from train_model import extract_features
import numpy as np
import os
import logging
import soundfile as sf
import librosa
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_audio_file(file_path):
    """Verify if the audio file is valid and can be processed."""
    try:
        logger.info(f"Starting audio file verification for: {file_path}")
        
        # Check basic file properties
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size} bytes")
        if file_size == 0:
            logger.error("File is empty")
            return False
            
        # Try reading with soundfile
        try:
            logger.info("Attempting to read with soundfile...")
            with sf.SoundFile(file_path) as audio_file:
                logger.info(f"SoundFile success - Sample rate: {audio_file.samplerate}Hz, "
                          f"Channels: {audio_file.channels}, Frames: {audio_file.frames}")
                if audio_file.frames == 0:
                    logger.error("Audio file has no frames")
                    return False
        except Exception as e:
            logger.warning(f"SoundFile read failed: {str(e)}\n{traceback.format_exc()}")
            # Don't return False here, try librosa as backup
            
        # Try reading with librosa
        try:
            logger.info("Attempting to read with librosa...")
            y, sr = librosa.load(file_path, duration=1, sr=None)
            logger.info(f"Librosa success - Sample rate: {sr}Hz, Length: {len(y)} samples")
            if len(y) == 0:
                logger.error("Librosa loaded empty audio data")
                return False
            return True
        except Exception as e:
            logger.error(f"Librosa read failed: {str(e)}\n{traceback.format_exc()}")
            return False
            
    except Exception as e:
        logger.error(f"Error in verify_audio_file: {str(e)}\n{traceback.format_exc()}")
        return False

def predict_healing_music(audio_path):
    """
    Predict whether a music file is healing or not.
    Returns: probability of being healing music (0-1)
    """
    try:
        # Step 1: Verify audio file
        logger.info(f"Starting prediction process for: {audio_path}")
        if not verify_audio_file(audio_path):
            logger.error("Audio file verification failed")
            return None
            
        # Step 2: Load model and scaler
        try:
            logger.info("Loading model and scaler...")
            if not os.path.exists('model.joblib') or not os.path.exists('scaler.joblib'):
                logger.error("Model or scaler file not found")
                return None
                
            model = joblib.load('model.joblib')
            scaler = joblib.load('scaler.joblib')
            logger.info("Model and scaler loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model or scaler: {str(e)}\n{traceback.format_exc()}")
            return None
        
        # Step 3: Extract features
        try:
            logger.info("Starting feature extraction...")
            features = extract_features(audio_path)
            if features is None:
                logger.error("Feature extraction returned None")
                return None
                
            # Verify feature dimensions
            expected_features = 33  # Based on our feature extraction
            if len(features) != expected_features:
                logger.error(f"Incorrect number of features. Expected {expected_features}, got {len(features)}")
                return None
                
            logger.info(f"Successfully extracted {len(features)} features")
            logger.debug(f"Feature values: {features}")  # 添加特征值的详细日志
        except Exception as e:
            logger.error(f"Error during feature extraction: {str(e)}\n{traceback.format_exc()}")
            return None
        
        # Step 4: Scale features
        try:
            logger.info("Scaling features...")
            features_scaled = scaler.transform(features.reshape(1, -1))
            logger.info("Features scaled successfully")
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}\n{traceback.format_exc()}")
            return None
        
        # Step 5: Make prediction
        try:
            logger.info("Making prediction...")
            probability = model.predict_proba(features_scaled)[0][1]
            logger.info(f"Prediction successful: {probability:.2f}")
            return probability
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}\n{traceback.format_exc()}")
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}\n{traceback.format_exc()}")
        return None
