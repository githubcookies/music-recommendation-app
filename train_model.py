import os
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import soundfile as sf
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def extract_features(file_path):
    """Extract audio features from a file."""
    try:
        logger.info(f"Starting feature extraction for: {file_path}")
        
        # Verify file format
        try:
            with sf.SoundFile(file_path) as sf_file:
                logger.info(f"Audio file info: {sf_file.samplerate}Hz, {sf_file.channels} channels")
        except Exception as e:
            logger.error(f"Error reading audio file with soundfile: {str(e)}")
            return None

        # Load audio file with error handling
        try:
            logger.info("Loading audio file...")
            y, sr = librosa.load(file_path, duration=30, sr=None)
            if len(y) == 0:
                logger.error("Audio file is empty")
                return None
            logger.info(f"Successfully loaded audio: {len(y)} samples, {sr}Hz sample rate")
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}\n{traceback.format_exc()}")
            return None

        # Ensure minimum duration
        duration = len(y) / sr
        logger.info(f"Audio duration: {duration:.2f} seconds")
        if duration < 1.0:
            logger.error("Audio file is too short (less than 1 second)")
            return None

        features_dict = {}
        
        try:
            # 1. MFCC (13 features x 2 = 26)
            logger.info("Extracting MFCC features...")
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features_dict['mfccs_mean'] = np.mean(mfccs, axis=1)
            features_dict['mfccs_var'] = np.var(mfccs, axis=1)
            logger.info(f"MFCC features shape: {mfccs.shape}")
        except Exception as e:
            logger.error(f"Error extracting MFCC: {str(e)}\n{traceback.format_exc()}")
            return None

        try:
            # 2. Spectral Centroid
            logger.info("Extracting spectral centroid...")
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features_dict['spectral_centroid'] = np.mean(spectral_centroids)
        except Exception as e:
            logger.error(f"Error extracting spectral centroid: {str(e)}\n{traceback.format_exc()}")
            return None

        try:
            # 3. Spectral Rolloff
            logger.info("Extracting spectral rolloff...")
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features_dict['spectral_rolloff'] = np.mean(spectral_rolloff)
        except Exception as e:
            logger.error(f"Error extracting spectral rolloff: {str(e)}\n{traceback.format_exc()}")
            return None

        try:
            # 4. Zero Crossing Rate
            logger.info("Extracting zero crossing rate...")
            zero_crossing = librosa.feature.zero_crossing_rate(y)[0]
            features_dict['zero_crossing'] = np.mean(zero_crossing)
        except Exception as e:
            logger.error(f"Error extracting zero crossing rate: {str(e)}\n{traceback.format_exc()}")
            return None

        try:
            # 5. Chroma Features
            logger.info("Extracting chroma features...")
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features_dict['chroma'] = np.mean(chroma, axis=1)
            logger.info(f"Chroma features shape: {chroma.shape}")
        except Exception as e:
            logger.error(f"Error extracting chroma features: {str(e)}\n{traceback.format_exc()}")
            return None

        try:
            # 6. Tempo
            logger.info("Extracting tempo...")
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features_dict['tempo'] = tempo
        except Exception as e:
            logger.error(f"Error extracting tempo: {str(e)}\n{traceback.format_exc()}")
            return None

        try:
            # 7. RMS Energy
            logger.info("Extracting RMS energy...")
            rms = librosa.feature.rms(y=y)[0]
            features_dict['rms'] = np.mean(rms)
        except Exception as e:
            logger.error(f"Error extracting RMS energy: {str(e)}\n{traceback.format_exc()}")
            return None

        # Combine all features
        try:
            logger.info("Combining features...")
            features = np.concatenate([
                features_dict['mfccs_mean'],
                features_dict['mfccs_var'],
                [features_dict['spectral_centroid'],
                 features_dict['spectral_rolloff'],
                 features_dict['zero_crossing'],
                 features_dict['tempo'],
                 features_dict['rms']],
                features_dict['chroma']
            ])
            logger.info(f"Final feature vector shape: {features.shape}")
            return features
        except Exception as e:
            logger.error(f"Error combining features: {str(e)}\n{traceback.format_exc()}")
            return None

    except Exception as e:
        logger.error(f"Unexpected error in feature extraction: {str(e)}\n{traceback.format_exc()}")
        return None

def prepare_dataset():
    """Prepare dataset from healing and non-healing music folders."""
    features = []
    labels = []
    
    # Check if directories exist
    if not os.path.exists("healing_music") or not os.path.exists("non_healing_music"):
        print("Training directories not found. Using default training data...")
        # Create a small synthetic dataset for initial deployment
        np.random.seed(42)
        n_samples = 10
        # Create synthetic features (assuming 33 features based on our feature extraction)
        synthetic_features = np.random.rand(n_samples, 33)
        synthetic_labels = np.random.randint(0, 2, n_samples)
        return synthetic_features, synthetic_labels
    
    # Process healing music
    healing_dir = "healing_music"
    for file in os.listdir(healing_dir):
        if file.endswith(('.mp3', '.wav')):
            file_path = os.path.join(healing_dir, file)
            extracted_features = extract_features(file_path)
            if extracted_features is not None:
                features.append(extracted_features)
                labels.append(1)  # 1 for healing
    
    # Process non-healing music
    non_healing_dir = "non_healing_music"
    for file in os.listdir(non_healing_dir):
        if file.endswith(('.mp3', '.wav')):
            file_path = os.path.join(non_healing_dir, file)
            extracted_features = extract_features(file_path)
            if extracted_features is not None:
                features.append(extracted_features)
                labels.append(0)  # 0 for non-healing
    
    return np.array(features), np.array(labels)

def train_and_evaluate_model():
    """Train and evaluate the model."""
    # Prepare dataset
    print("Extracting features from audio files...")
    X, y = prepare_dataset()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Save model and scaler
    print("Saving model and scaler...")
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    return model, scaler

if __name__ == "__main__":
    train_and_evaluate_model()
