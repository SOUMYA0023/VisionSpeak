import streamlit as st
from PIL import Image
import os
import sys

# Add project root to path to allow direct imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from inference.generate_caption import generate_caption

# --- Streamlit App Configuration --- #
st.set_page_config(page_title="Image Caption Generator", layout="centered")

st.title("AI Image Caption Generator")
st.write("Upload an image and our AI will generate a caption for you. This project uses a CNN-LSTM architecture (ResNet-50 + LSTM) trained on the Flickr8k dataset.")

# --- Model and Vocab Paths --- #
# These paths are relative to the project root where you run `streamlit run app.py`
MODEL_PATH = 'models/weights/caption-model-5.pth' # Update if you use a different epoch
VOCAB_PATH = 'data/processed/vocab.pkl'

# --- Image Upload and Caption Generation --- #
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    st.write("") # Add a little space

    # Generate caption on button click
    if st.button('Generate Caption'):
        # Check if model and vocab files exist
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
            st.error("Error: Model or vocabulary file not found. Please make sure the paths are correct and you have trained the model.")
        else:
            with st.spinner('Generating caption...'):
                try:
                    # To pass the image to the function, we need to save it temporarily
                    temp_image_path = os.path.join("data", "temp_uploaded_image.jpg")
                    image.save(temp_image_path)

                    # Generate the caption
                    caption = generate_caption(temp_image_path, MODEL_PATH, VOCAB_PATH)
                    
                    # Display the caption
                    st.success('**Generated Caption:**')
                    st.write(f'### "{caption.capitalize()}"')

                    # Clean up the temporary file
                    os.remove(temp_image_path)

                except Exception as e:
                    st.error(f"An error occurred during caption generation: {e}")

# --- Instructions and Project Info --- #
st.sidebar.header("About the Project")
st.sidebar.info(
    "**Tech Stack:**\n" 
    "- Python 3.10\n" 
    "- PyTorch\n" 
    "- NLTK\n" 
    "- Streamlit\n\n" 
    "**Model:**\n" 
    "- **Encoder:** Pretrained ResNet-50\n" 
    "- **Decoder:** LSTM Network\n\n" 
    "For more details, check out the `README.md` on the project's GitHub page."
)
