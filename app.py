import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
import base64
import requests
from edge_detection import process_image
import matplotlib.pyplot as plt
import uuid

def get_image_base64(img_array):
    img = Image.fromarray(img_array)
    if img.mode=="RGBA":
        img = img.convert("RGB")
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def show_blur_overlay():
    st.markdown(
        """
        <style>
        .blur-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            width: 100vw; height: 100vh;
            background: rgba(30, 30, 30, 0.5);
            backdrop-filter: blur(8px);
            z-index: 9999;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            transition: opacity 0.5s;
        }
        .single-spinner {
            border: 5px solid rgba(224,224,224,0.7);
            border-top: 5px solid rgba(52,152,219,0.9);
            border-radius: 50%;
            width: 48px;
            height: 48px;
            animation: spin 1s linear infinite;
            margin-bottom: 24px;
            opacity: 0.85;
        }
        @keyframes spin {
            0% { transform: rotate(0deg);}
            100% { transform: rotate(360deg);}
        }
        .processing-text {
            color: #fff;
            font-size: 1.3em;
            font-family: 'Fira Mono', 'Consolas', monospace;
            margin: 0 auto;
            letter-spacing: .10em;
            display: flex;
            align-items: center;
        }
        .dot {
            opacity: 0;
            animation: blink 1.2s infinite;
        }
        .dot:nth-child(1) { animation-delay: 0s; }
        .dot:nth-child(2) { animation-delay: 0.2s; }
        .dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes blink {
            0%, 80%, 100% { opacity: 0; }
            40% { opacity: 1; }
        }
        </style>
        <div class="blur-overlay">
            <div class="single-spinner"></div>
            <div class="processing-text">
                Processing Image<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(page_title="Edge Detection", layout="wide")

# Add custom CSS to control image sizes
st.markdown("""
    <style>
    .stImage {
        max-width: 100%;
        height: auto;
    }
    .uploaded-image {
        max-width: 600px;
        margin: 0 auto;
    }
    .result-image {
        max-width: 600px;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Edge Detection using Pb-lite")
st.markdown("---")

# Add a session state variable for the uploader key
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
# Add a session state variable to track if an image has been processed
if 'has_processed' not in st.session_state:
    st.session_state.has_processed = False

# File uploader
uploaded_file = None
if not st.session_state.has_processed or st.session_state.processed_results is None:
    st.markdown("#### Choose any image of your choice to get started")
    uploaded_file = st.file_uploader("Image Upload", type=["jpg", "jpeg", "png", "webp"], key=st.session_state.uploader_key, label_visibility="collapsed")

# Add a text input for image URL
image_url = None
if not st.session_state.has_processed or st.session_state.processed_results is None:
    st.markdown("#### Or paste an image URL")
    image_url = st.text_input("Image URL", placeholder="Enter URL (jpg, jpeg, png, webp)", label_visibility="collapsed")

# Initialize session state for storing the uploaded image and overlay state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = None
if 'show_blur' not in st.session_state:
    st.session_state.show_blur = False

# Store the uploaded image in session state
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif image_url:
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"Could not load image from URL: {e}")

if image is not None:
    image = np.array(image)
    st.session_state.uploaded_image = image
    
    if st.session_state.processed_results is None:
        img_base64 = get_image_base64(image)
        st.markdown(
            f'''
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 60vh;">
                <h3 style="margin-bottom: 20px;">Uploaded Image</h3>
                <img src="data:image/jpeg;base64,{img_base64}" width="600" height="400" style="border-radius: 10px; object-fit: cover; margin-bottom: 20px;" />
            </div>
            ''',
            unsafe_allow_html=True
        )
        # Center the button using columns
        button_col1, button_col2, button_col3 = st.columns([4, 1, 4])
        with button_col2:
            if st.button("Process Image", type="primary", use_container_width=True):
                st.session_state.show_blur = True
                st.rerun()

# Show the blur overlay and process the image if requested
if st.session_state.get('show_blur', False):
    show_blur_overlay()
    # Actually process the image
    image = st.session_state.uploaded_image
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    unique_id = str(uuid.uuid4())
    temp_path = os.path.join(temp_dir, f"temp_image_{unique_id}.jpg")
    img_to_save = Image.fromarray(image)
    if img_to_save.mode == "RGBA":
        img_to_save = img_to_save.convert("RGB")
    img_to_save.save(temp_path)

    results = process_image(temp_path)
    st.session_state.processed_results = results
    st.session_state.has_processed = True
    if os.path.exists(temp_path):
        os.remove(temp_path)
    st.session_state.show_blur = False
    st.rerun()

# Display results if they exist
if st.session_state.processed_results is not None:
    results = st.session_state.processed_results
    
    # Create two columns for the uploaded image and edge detection result
    result_col1, result_col2 = st.columns([1, 1])
    
    with result_col1:
        st.subheader("Original Image")
        st.image(st.session_state.uploaded_image, use_container_width=True)
    
    with result_col2:
        st.subheader("Edge Detection Result")
        st.image(results['final_output'], use_container_width=True)
    
    # Display intermediate results in a collapsible section
    with st.expander("View Intermediate Results", expanded=False):
        st.subheader("Intermediate Results")
        cols = st.columns(3)
        
        with cols[0]:
            st.write("Texture Map")
            st.image(results['texture_map'], use_container_width=True)
            
            st.write("Texture Gradient")
            st.image(results['texture_gradient'], use_container_width=True)
        
        with cols[1]:
            st.write("Brightness Map")
            st.image(results['brightness_map'], use_container_width=True)
            
            st.write("Brightness Gradient")
            st.image(results['brightness_gradient'], use_container_width=True)
        
        with cols[2]:
            st.write("Color Map")
            st.image(results['color_map'], use_container_width=True)
            
            st.write("Color Gradient")
            st.image(results['color_gradient'], use_container_width=True)

    # Add a button to process another image
    st.markdown("---")
    if st.button("Process Another Image"):
        st.session_state.processed_results = None
        st.session_state.uploaded_image = None
        st.session_state.show_blur = False
        st.session_state.uploader_key += 1  # Reset uploader widget
        st.session_state.has_processed = False
        st.rerun()

elif uploaded_file is None and not image_url:
    st.info("Please upload an image to begin edge detection or paste a URL above.")
