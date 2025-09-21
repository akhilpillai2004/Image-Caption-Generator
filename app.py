import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from PIL import Image

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="📸",
    layout="centered"
)

# --- MODEL AND TOKENIZER LOADING ---
# Use Streamlit's caching to load models only once
@st.cache_resource
def load_assets():
    """Loads the trained models and tokenizer from disk."""
    try:
        # Update these paths to where you've stored your model files
        feature_extractor = load_model("feature_extractor.keras", compile=False)
        caption_model = load_model("image_caption_model.keras", compile=False)
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return feature_extractor, caption_model, tokenizer
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None, None

# --- CAPTION GENERATION LOGIC ---
def generate_caption(feature_extractor, caption_model, tokenizer, image, max_length):
    """Generates a caption for a given image."""
    # Preprocess the image
    img = image.resize((224, 224))
    img = img.convert("RGB")
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Extract features
    photo_feature = feature_extractor.predict(img_array, verbose=0)
    
    # Generate caption
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([photo_feature, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
            
    return in_text.replace("startseq", "").replace("endseq", "").strip()

# --- STREAMLIT APP INTERFACE ---
st.title("📸 Image Caption Generator")
st.info("Upload an image to generate a descriptive caption using a deep learning model.")

# Load models
feature_extractor, caption_model, tokenizer = load_assets()

if feature_extractor is not None:
    uploaded_image = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        # Display the uploaded image
        st.image(image, caption="Your Uploaded Image", use_column_width=True)
        
        # Generate caption when button is clicked
        if st.button("Generate Caption", type="primary"):
            with st.spinner("Generating caption, please wait..."):
                max_length = 34 
                caption = generate_caption(feature_extractor, caption_model, tokenizer, image, max_length)
                st.success("Caption Generated!")
                st.subheader("Predicted Caption:")
                st.write(f"> {caption.capitalize()}")
else:
    st.error("Model assets could not be loaded. Please check the file paths and integrity.")