import streamlit as st
import os
from predict import predict_healing_music
import tempfile
import train_model
import logging
import io

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建一个 StringIO 对象来捕获日志
log_stream = io.StringIO()
handler = logging.StreamHandler(log_stream)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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

# 添加调试模式切换
debug_mode = st.sidebar.checkbox("Enable Debug Mode")

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
        # 清除之前的日志
        log_stream.seek(0)
        log_stream.truncate()
        
        # Update status
        status_text.text("Analyzing your music...")
        progress_bar.progress(30)
        
        # 记录文件信息
        file_size = len(uploaded_file.getvalue())
        logger.info(f"Processing uploaded file: {uploaded_file.name} (size: {file_size} bytes)")
        
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
            st.error("Sorry, there was an error analyzing your music file.")
            st.write("Please check the following:")
            st.write("1. Make sure the file is a valid audio file (MP3 or WAV)")
            st.write("2. The file is not corrupted or empty")
            st.write("3. The file format is supported (try converting to WAV if needed)")
            st.write("4. The audio duration is at least 1 second")
            
            # Add technical details in an expander
            with st.expander("Technical Details", expanded=True):
                st.write("The error could be due to one of the following:")
                st.write("- File format not recognized by the audio processing libraries")
                st.write("- Unsupported audio codec or compression")
                st.write("- File corruption during upload")
                st.write("- Insufficient audio duration for feature extraction")
                
                # 显示详细的技术日志
                st.subheader("Technical Log")
                st.code(log_stream.getvalue())
                
            st.write("Try uploading a different file or convert your file to WAV format using an audio converter.")
            
        # Clean up
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # 在调试模式下显示完整日志
        if debug_mode:
            st.sidebar.subheader("Debug Log")
            st.sidebar.code(log_stream.getvalue())
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.exception("Unexpected error")
        if debug_mode:
            st.sidebar.subheader("Error Log")
            st.sidebar.code(log_stream.getvalue())
        
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
