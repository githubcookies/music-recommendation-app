import joblib
from train_model import extract_features
import numpy as np
import os
import logging
import soundfile as sf
import librosa
import traceback

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def verify_audio_file(file_path):
    """Verify if the audio file is valid and can be processed."""
    try:
        logger.debug(f"Starting audio file verification for: {file_path}")
        
        # Check basic file properties
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        file_size = os.path.getsize(file_path)
        logger.debug(f"File size: {file_size} bytes")
        if file_size == 0:
            logger.error("File is empty")
            return False
            
        # Try reading with librosa first (more robust)
        try:
            logger.debug("Attempting to read with librosa...")
            y, sr = librosa.load(file_path, duration=5, sr=None)  # 增加到5秒以获取更好的特征
            logger.debug(f"Librosa success - Sample rate: {sr}Hz, Length: {len(y)} samples")
            if len(y) == 0:
                logger.error("Librosa loaded empty audio data")
                return False
            return True
        except Exception as e:
            logger.warning(f"Librosa read failed with error: {str(e)}")
            
            # Try soundfile as backup
            try:
                logger.debug("Attempting to read with soundfile...")
                with sf.SoundFile(file_path) as audio_file:
                    logger.debug(f"SoundFile success - Sample rate: {audio_file.samplerate}Hz, "
                              f"Channels: {audio_file.channels}, Frames: {audio_file.frames}")
                    if audio_file.frames == 0:
                        logger.error("Audio file has no frames")
                        return False
                    return True
            except Exception as sf_error:
                logger.error(f"Both Librosa and SoundFile failed to read the audio file")
                logger.debug(f"SoundFile error: {str(sf_error)}")
                return False
            
    except Exception as e:
        logger.error(f"Error in verify_audio_file: {str(e)}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return False

def predict_healing_music(audio_path):
    """
    Predict whether a music file is healing or not.
    Returns: probability of being healing music (0-1)
    """
    try:
        # Step 1: Verify audio file
        print(f"[DEBUG] Starting prediction process for: {audio_path}")
        logger.info(f"Starting prediction process for: {audio_path}")
        if not verify_audio_file(audio_path):
            print("[ERROR] Audio file verification failed")
            logger.error("Audio file verification failed")
            return None
            
        # Step 2: Load model and scaler
        try:
            print("[DEBUG] Loading model and scaler...")
            logger.info("Loading model and scaler...")
            if not os.path.exists('model.joblib') or not os.path.exists('scaler.joblib'):
                print("[ERROR] Model or scaler file not found")
                logger.error("Model or scaler file not found")
                return None
                
            model = joblib.load('model.joblib')
            scaler = joblib.load('scaler.joblib')
            print("[DEBUG] Model and scaler loaded successfully")
            logger.info("Model and scaler loaded successfully")
        except Exception as e:
            print(f"[ERROR] Error loading model or scaler: {str(e)}")
            logger.error(f"Error loading model or scaler: {str(e)}")
            return None
        
        # Step 3: Extract features
        try:
            print("[DEBUG] Starting feature extraction...")
            logger.info("Starting feature extraction...")
            features = extract_features(audio_path)
            if features is None:
                print("[ERROR] Feature extraction returned None")
                logger.error("Feature extraction returned None")
                return None
                
            # Verify feature dimensions
            expected_features = 38  # Updated to match our current feature extraction (26 MFCC + 12 Chroma)
            if len(features) != expected_features:
                print(f"[ERROR] Incorrect number of features. Expected {expected_features}, got {len(features)}")
                logger.error(f"Incorrect number of features. Expected {expected_features}, got {len(features)}")
                return None
                
            print(f"[DEBUG] Successfully extracted {len(features)} features")
            logger.info(f"Successfully extracted {len(features)} features")
            logger.debug(f"Feature values: {features}")
        except Exception as e:
            print(f"[ERROR] Error during feature extraction: {str(e)}")
            logger.error(f"Error during feature extraction: {str(e)}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None
        
        # Step 4: Scale features
        try:
            print("[DEBUG] Scaling features...")
            logger.info("Scaling features...")
            features_scaled = scaler.transform(features.reshape(1, -1))
            print("[DEBUG] Features scaled successfully")
            logger.info("Features scaled successfully")
        except Exception as e:
            print(f"[ERROR] Error scaling features: {str(e)}")
            logger.error(f"Error scaling features: {str(e)}")
            return None
        
        # Step 5: Make prediction
        try:
            print("[DEBUG] Making prediction...")
            logger.info("Making prediction...")
            probability = model.predict_proba(features_scaled)[0][1]
            print(f"[DEBUG] Prediction successful: {probability:.2f}")
            logger.info(f"Prediction successful: {probability:.2f}")
            return probability
        except Exception as e:
            print(f"[ERROR] Error making prediction: {str(e)}")
            logger.error(f"Error making prediction: {str(e)}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Unexpected error during prediction: {str(e)}")
        logger.error(f"Unexpected error during prediction: {str(e)}")
        return None
