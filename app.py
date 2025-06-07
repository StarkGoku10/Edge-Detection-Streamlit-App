import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import base64
import requests
import time
from edge_detection import process_image 
import uuid # For generating unique filenames

LOCK_FILE_PATH = "app.lock"

def get_image_base64(img_array_or_pil_img):
    """Converts a NumPy array or PIL Image to a base64 encoded JPEG string."""
    if isinstance(img_array_or_pil_img, np.ndarray):
        img = Image.fromarray(img_array_or_pil_img.astype(np.uint8))
    elif isinstance(img_array_or_pil_img, Image.Image):
        img = img_array_or_pil_img
    else:
        raise ValueError("Input must be a NumPy array or PIL Image.")

    if img.mode == "RGBA":
        # Ensure it's RGB for JPEG
        img = img.convert("RGB")
    elif img.mode == "L": # Grayscale
        # Convert to RGB if it's grayscale for consistent display
        img = img.convert("RGB")

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def show_blur_overlay():
    """Displays a processing overlay with a spinner."""
    st.markdown(
        """
        <style>
        .blur-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            width: 100vw; height: 100vh;
            background: rgba(30, 30, 30, 0.5);
            backdrop-filter: blur(8px); /* Standard blur */
            -webkit-backdrop-filter: blur(8px); /* Safari */
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
            font-family: 'Fira Mono', 'Consolas', monospace; /* Monospaced font for dots */
            margin: 0 auto;
            letter-spacing: .10em;
            display: flex;
            align-items: center;
        }
        .dot { /* Animated dots for processing text */
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

# --- App Initialization ---
st.set_page_config(page_title="Edge Detection Pro", layout="wide")

# Custom CSS for image display and layout
st.markdown("""
    <style>
    /* Ensure Streamlit images are responsive */
    .stImage > img {
        max-width: 100%; /* Ensure images scale down */
        height: auto;    /* Maintain aspect ratio */
        object-fit: contain; /* Ensure the whole image is visible */
        border-radius: 8px; /* Slightly rounded corners for images */
    }
    .uploaded-image-container, .result-image-container {
        display: flex;
        justify-content: center; /* Center images horizontally */
        align-items: center;     /* Center images vertically if needed */
        padding: 10px;
    }
    .uploaded-image-container img, .result-image-container img {
        max-width: 500px; /* Max width for uploaded/result images */
        max-height: 500px; /* Max height to prevent overly large images */
        border: 1px solid #ddd; /* Subtle border */
    }
    .main-title { /* Custom title style */
        text-align: center;
        color: #2c3e50; /* Darker color */
        padding-bottom: 10px;
        border-bottom: 2px solid #3498db; /* Accent color border */
    }
    .section-header { /* Style for subheaders */
        color: #34495e;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    /* Style for the button */
    div.stButton > button {
        border-radius: 20px; /* More rounded buttons */
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Edge Detection using Pb-lite Algorithm</h1>", unsafe_allow_html=True)
st.markdown("---")

# Initialize session state variables
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'has_processed' not in st.session_state:
    st.session_state.has_processed = False
if 'uploaded_image_data' not in st.session_state: # To store the PIL Image object
    st.session_state.uploaded_image_data = None
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = None
if 'show_blur' not in st.session_state:
    st.session_state.show_blur = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = None


# --- Image Input Section ---
if not st.session_state.has_processed:
    st.markdown("<h4 class='section-header'>Upload an Image or Provide a URL to get started</h4>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("<p class='input-label'>Browse files</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "### Browse files",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed" # Unique key for reset
    )
    
    st.markdown("<p style='text-align: center; margin: 0;'> or</p>", unsafe_allow_html=True)
    st.markdown("<p class='input-label'>Enter URL</p>", unsafe_allow_html=True)
    image_url = st.text_input(
        "Enter URL",
        placeholder="https://worldarchitecture.org/cdnimgfiles/extuploadc/lotustemple16b-1-.jpg",
        label_visibility="collapsed",
    )
    
    pil_image = None
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file)
            # Clear previous errors
            st.session_state.error_message = None 
        except Exception as e:
            st.error(f"Error opening uploaded file: {e}")
            st.session_state.error_message = f"Could not load image from uploaded file: {e}"
            pil_image = None
            
    elif image_url:
        try:
            # Added timeout to handle slow responses
            response = requests.get(image_url, timeout=10)
            # Raises an HTTPError for bad responses (4XX or 5XX)
            response.raise_for_status() 
            pil_image = Image.open(io.BytesIO(response.content))
            # Clear previous errors
            st.session_state.error_message = None 
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching image from URL: {e}")
            st.session_state.error_message = f"Could not load image from URL: {e}"
            pil_image = None
            # Catch other PIL errors
        except Exception as e: 
            st.error(f"Error opening image from URL: {e}")
            st.session_state.error_message = f"Could not process image from URL: {e}"
            pil_image = None

    if pil_image is not None:
        st.session_state.uploaded_image_data = pil_image # Store PIL image
        
        # Display uploaded image preview
        st.markdown("<h3 class='section-header' style='text-align: center;'>Uploaded Image Preview</h3>", unsafe_allow_html=True)
        # Ensure RGB for display
        img_array_for_display = np.array(pil_image.convert("RGB")) 
        # center the image and button
        prev_col1, prev_col_main, prev_col3 = st.columns([1,2,1])
        with prev_col_main:
            st.image(img_array_for_display, use_container_width=True, caption="Your Uploaded Image")

            # lock check mechanism & check if a process is already running
            is_locked = os.path.exists(LOCK_FILE_PATH)
            if is_locked:
                # check for stale lock file older than 5 minutes
                try:  
                    if (time.time() - os.path.getmtime(LOCK_FILE_PATH)) > 300:
                        os.remove(LOCK_FILE_PATH)
                        is_locked=False
                except FileNotFoundError:
                    is_locked = False 

            if is_locked:
                st.warning("⏳ The Magic Wand✨ is currently in use! Please hold on, your patience will be rewarded. 🙏")
                if st.button("🔄 Check if it is your turn to create Magic"):
                    st.rerun()
            else:
                if st.button("✨ Unveil the Edges!", type="primary", use_container_width=True):
                    st.session_state.show_blur = True
                    # Clear previous errors before processing
                    st.session_state.error_message = None 
                    st.rerun()

# --- Processing Logic ---
if st.session_state.get('show_blur', False) and st.session_state.uploaded_image_data is not None:
    # creation of a lock file to signal that the process is running/has started
    with open(LOCK_FILE_PATH, "w") as f:
        f.write(str(int(time.time())))
    show_blur_overlay()
    
    # temporary directory if it doesn't exist
    temp_dir = "temp_streamlit_images" 
    os.makedirs(temp_dir, exist_ok=True)
    
    # unique filename for the temporary image
    unique_filename = f"{uuid.uuid4()}.jpg"
    temp_path = os.path.join(temp_dir, unique_filename)
    
    processing_success = False
    try:
        # Save the PIL image to the temporary path
        # Convert to RGB before saving to ensure compatibility with process_image if it expects 3 channels
        img_to_save = st.session_state.uploaded_image_data.convert("RGB")
        img_to_save.save(temp_path, format="JPEG")

        # Call the image processing function
        results = process_image(temp_path) # process_image expects a path
        st.session_state.processed_results = results
        st.session_state.has_processed = True
        processing_success = True
        # Clear error on success
        st.session_state.error_message = None 
        
    except Exception as e:
        st.session_state.error_message = f"An error occurred during image processing: {str(e)}"
        # Reset states to allow user to try again
        st.session_state.processed_results = None
        st.session_state.has_processed = False
        # Keep uploaded_image_data so the user does not have to re-upload if it was a processing error
        
    finally:
        # cleanup the lock file
        if os.path.exists(LOCK_FILE_PATH):
            try:
                os.remove(LOCK_FILE_PATH)
            except Exception as e_lock: 
                print(f"Error removing lock file {LOCK_FILE_PATH}: {e_lock}")

        # Clean up the temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e_remove:
                # Log to console
                print(f"Error removing temporary file {temp_path}: {e_remove}") 
        
        st.session_state.show_blur = False # Hide overlay
        st.rerun() # Rerun to update UI based on success/failure


# --- Display Error if any ---
if st.session_state.error_message and not st.session_state.has_processed: # Show error if processing failed or input failed
    st.error(st.session_state.error_message)
    # Offer a way to clear the error and try again
    if st.button("Try Uploading Again"):
        st.session_state.error_message = None
        st.session_state.uploader_key += 1
        st.session_state.uploaded_image_data = None
        st.session_state.processed_results = None
        st.session_state.has_processed = False
        st.rerun()


# --- Display Results Section ---
if st.session_state.has_processed and st.session_state.processed_results is not None:
    results = st.session_state.processed_results
    
    st.markdown("<h2 class='section-header' style='text-align: center;'>Edge Detection Results</h2>", unsafe_allow_html=True)
    
    # Display Original and Final Output side-by-side
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.markdown("<h3 class='section-header'>Original Image</h3>", unsafe_allow_html=True)
        # Display the initially uploaded image (PIL object)
        st.image(st.session_state.uploaded_image_data, use_container_width=True) 
    
    with res_col2:
        st.markdown("<h3 class='section-header'>Edge Detection (Pb-lite)</h3>", unsafe_allow_html=True)
        st.image(results['final_output'], use_container_width=True)
    
    st.markdown("---")
    
    # Display intermediate results in a collapsible section
    with st.expander("🔬 View Intermediate Results", expanded=False):
        st.subheader("Intermediate Maps & Gradients")
        
        # Determine number of columns needed based on available results
        # Max 3 items per row for better layout
        intermediate_items = {
            "Texture Map": results.get('texture_map'),
            "Texture Gradient": results.get('texture_gradient'),
            "Brightness Map": results.get('brightness_map'),
            "Brightness Gradient": results.get('brightness_gradient'),
            "Color Map": results.get('color_map'),
            "Color Gradient": results.get('color_gradient'),
            "Sobel Baseline": results.get('sobel_baseline'),
            "Canny Baseline": results.get('canny_baseline')
        }
        
        displayable_items = {k: v for k, v in intermediate_items.items() if v is not None}
        
        if displayable_items:
            num_cols = min(3, len(displayable_items))
            cols = st.columns(num_cols)
            col_idx = 0
            for title, img_data in displayable_items.items():
                with cols[col_idx % num_cols]:
                    st.markdown(f"**{title}**")
                    st.image(img_data, use_container_width=True)
                    col_idx += 1
        else:
            st.write("No intermediate results available.")

    # Button to process another image
    st.markdown("---")
    if st.button("🔄 Process Another Image", use_container_width=True):
        # Reset all relevant session state variables
        st.session_state.processed_results = None
        # Clear stored PIL image
        st.session_state.uploaded_image_data = None 
        st.session_state.show_blur = False
        # Increment key to reset file_uploader
        st.session_state.uploader_key += 1 
        st.session_state.has_processed = False
        # Clear any previous errors
        st.session_state.error_message = None 
        st.rerun()

elif not st.session_state.uploaded_image_data and not st.session_state.has_processed and not st.session_state.error_message:
    st.info("👋 Welcome! Please upload an image or provide a URL to begin edge detection.")

