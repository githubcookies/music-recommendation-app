import streamlit as st
import os
from predict import predict_healing_music
import tempfile
import train_model
import logging
import io

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Healing Music Classifier",
    page_icon="🎵",
    layout="centered"
)

# Ensure model directory exists
model_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(model_dir, exist_ok=True)

# Model file paths
model_path = os.path.join(model_dir, "model.joblib")
scaler_path = os.path.join(model_dir, "scaler.joblib")

# Check if model files exist
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.info('First run: Training the model...')
    try:
        train_model.train_and_evaluate_model()
        st.success('Model training completed!')
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        st.error(f'Model training failed: {str(e)}')
        st.stop()

st.title("🎵 Healing Music Classifier")
st.write("""
Upload your music file, and AI will analyze its healing potential!
""")

# Add file size limit warning
st.write("Note: Please upload files smaller than 10MB for best results.")

# Add file upload component with more specific instructions
uploaded_file = st.file_uploader(
    "Choose an audio file...", 
    type=['mp3', 'wav'],
    help="Supported formats: MP3, WAV. Recommended duration: 5-30 seconds."
)

if uploaded_file is not None:
    # Check file size
    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Convert to MB
    if file_size > 10:
        st.warning(f"File size ({file_size:.1f}MB) is larger than recommended (10MB). This might affect processing time.")
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            # Write uploaded file content
            file_content = uploaded_file.getvalue()
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        # Update status
        status_text.text("Analyzing music...")
        progress_bar.progress(30)
        
        # Log file information
        logger.info(f"Processing uploaded file: {uploaded_file.name} (size: {len(file_content)} bytes)")
        
        # Make prediction
        healing_probability = predict_healing_music(tmp_file_path)
        progress_bar.progress(90)
        
        if healing_probability is not None:
            # Display results
            st.subheader("Analysis Results")
            
            # Create visualization progress bar
            healing_percentage = healing_probability * 100
            st.progress(healing_probability)
            
            # Display percentage
            st.write(f"Healing Index: {healing_percentage:.1f}%")
            
            # Provide explanation
            if healing_percentage >= 75:
                st.success("This music has strong healing properties! 🌟")
            elif healing_percentage >= 50:
                st.info("This music has moderate healing effects. ✨")
            else:
                st.warning("This music has limited healing potential. 🎵")
        else:
            st.error("Sorry, an error occurred while analyzing the music file.")
            st.write("Please check the following:")
            st.write("1. Ensure the file is a valid audio file (MP3 or WAV format)")
            st.write("2. File is not corrupted and not empty")
            st.write("3. Audio duration is at least 5 seconds")
            st.write("4. File size is preferably under 10MB")
            logger.error(f"Prediction failed for file: {uploaded_file.name}")
            
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.exception(f"Unexpected error processing file: {uploaded_file.name}")
        
    finally:
        # Clean up temporary file
        try:
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        except Exception as e:
            logger.error(f"Failed to clean up temporary file: {str(e)}")
        
        # Complete progress bar
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
