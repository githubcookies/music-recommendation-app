# Healing Music Classifier

This project uses machine learning to classify whether a piece of music has healing properties or not. It analyzes various audio features including MFCC, spectral characteristics, rhythm, and harmonic content to make predictions.

## Features

- Audio feature extraction using librosa
- Machine learning classification using Random Forest
- Web interface for easy music upload and analysis
- Visual results with healing probability score
- Cross-validation for model evaluation

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. First, train the model:
```bash
python train_model.py
```

2. Run the web application:
```bash
streamlit run app.py
```

3. Open your browser and upload a music file to analyze

## Project Structure

- `train_model.py`: Feature extraction and model training
- `predict.py`: Prediction functionality
- `app.py`: Streamlit web interface
- `requirements.txt`: Project dependencies
- `model.joblib`: Trained model (generated after training)
- `scaler.joblib`: Feature scaler (generated after training)

## Technical Details

The classifier uses the following features:
- Mel-frequency cepstral coefficients (MFCC)
- Spectral centroid
- Spectral rolloff
- Zero crossing rate
- Chroma features
- Tempo
- RMS energy

## Deployment

### Local Deployment
1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the web application:
```bash
streamlit run app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Fork this repository to your GitHub account
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app" and select this repository
5. Select the main branch and the app.py file
6. Click "Deploy"

Note: Make sure to include some sample music files in the `healing_music` and `non_healing_music` folders for training the model.

## License

MIT License

## Contributing

Feel free to open issues and pull requests!
