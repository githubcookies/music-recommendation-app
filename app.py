import streamlit as st
import os
from predict import predict_healing_music
import tempfile

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
            st.metric(
                label="Healing Score",
                value=f"{healing_percentage:.1f}%"
            )
            
            # Provide interpretation
            if healing_probability > 0.7:
                st.success("This music appears to be very healing! 🌟")
            elif healing_probability > 0.5:
                st.info("This music has some healing qualities. ✨")
            else:
                st.warning("This music might not be particularly healing. 🤔")
                
            # Additional insights
            st.write("---")
            st.write("Remember that music perception is subjective, and these results are based on patterns learned from our dataset.")
            
        else:
            st.error("Sorry, there was an error analyzing your music file. Please try another file.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
    finally:
        # Clean up the temporary file
        try:
            os.unlink(tmp_file_path)
        except:
            pass
        
        # Complete the progress bar
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
