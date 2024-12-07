import streamlit as st
import os
from predict import predict_healing_music
import tempfile
import train_model

# 检查模型文件是否存在，如果不存在则训练
if not os.path.exists('model.joblib') or not os.path.exists('scaler.joblib'):
    st.info('First time running: Training the model...')
    train_model.train_and_save_model()
    st.success('Model training completed!')

st.set_page_config(
    page_title="Healing Music Classifier",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Healing Music Classifier")
st.write("""
Upload your music file and our AI will analyze whether it's healing music or not!
""")

uploaded_file = st.file_uploader("Choose an audio file...", type=['mp3', 'wav'])

if uploaded_file is not None:
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Update status
        status_text.text("Analyzing your music...")
        progress_bar.progress(30)
        
        # Make prediction
        healing_probability = predict_healing_music(tmp_file_path)
        progress_bar.progress(90)
        
        if healing_probability is not None:
            # Display results
            st.subheader("Analysis Results")
            
            # Create a visual meter
            healing_percentage = healing_probability * 100
            st.progress(healing_probability)
            
            # Display the percentage
            st.write(f"Healing Score: {healing_percentage:.1f}%")
            
            # Provide interpretation
            if healing_percentage >= 75:
                st.success("This music has strong healing properties! 🌟")
            elif healing_percentage >= 50:
                st.info("This music has moderate healing properties. ✨")
            else:
                st.warning("This music has limited healing properties. 🎵")
        else:
            st.error("Sorry, there was an error analyzing your music file. Please check the following:")
            st.write("1. Make sure the file is a valid audio file (MP3 or WAV)")
            st.write("2. The file is not corrupted or empty")
            st.write("3. Try uploading a different file")
            
        # Clean up
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        
