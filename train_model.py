import os
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

def extract_features(file_path):
    """Extract audio features from a file."""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, duration=30)
        
        # Feature extraction
        # 1. MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_var = np.var(mfccs, axis=1)
        
        # 2. Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid_mean = np.mean(spectral_centroids)
        
        # 3. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        
        # 4. Zero Crossing Rate
        zero_crossing = librosa.feature.zero_crossing_rate(y)[0]
        zero_crossing_mean = np.mean(zero_crossing)
        
        # 5. Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # 6. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # 7. RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        
        # Combine all features
        features = np.concatenate([
            mfccs_mean, mfccs_var,
            [spectral_centroid_mean, spectral_rolloff_mean, 
             zero_crossing_mean, tempo, rms_mean],
            chroma_mean
        ])
        
        return features
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def prepare_dataset():
    """Prepare dataset from healing and non-healing music folders."""
    features = []
    labels = []
    
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
