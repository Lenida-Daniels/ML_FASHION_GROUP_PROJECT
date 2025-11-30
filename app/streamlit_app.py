import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import traceback
from dotenv import load_dotenv
from gemini_integration import FashionPriceAPI
from utils import get_class_names, preprocess_single_image_for_cnn

# Load environment variables
load_dotenv()
# Load model once
#model = load_trained_model("models/fashion_cnn.h5")
#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
 #              'Sandal', 'Shirt', 'Sneaker', 'Bag']

 # PAGE SETUP 
st.set_page_config(page_title="Fashion Classifier & Price Finder", layout="wide")

# CUSTOM STYLING 
st.markdown("""
    <style>
    /* Background gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to right, #1e1e2f, #252540);
        color: white;
    }

    /* Hide Streamlit branding */
    #MainMenu, header, footer {visibility: hidden;}

    /* Title style */
    h1 {
        text-align: center;
        color: #ffb6b9;
        font-size: 3em;
        margin-top: 30px;
        letter-spacing: 1px;
    }

    /* Subtitle style */
    p.subtitle {
        text-align: center;
        color: #d3d3d3;
        font-size: 1.2em;
        margin-bottom: 40px;
    }

    /* Upload box styling */
    [data-testid="stFileUploader"] {
        background-color: #2c2c3a;
        border: 1px solid #ffb6b9;
        border-radius: 12px;
        padding: 20px;
    }

    /* Text input */
    input[type="text"] {
        background-color: #2c2c3a;
        color: white;
        border: 1px solid #ffb6b9;
        border-radius: 8px;
        padding: 10px;
    }

    /* Button styling */
    div.stButton > button {
        background-color: #ffb6b9;
        color: black;
        border-radius: 10px;
        height: 3em;
        width: 50%;
        margin: 0 auto;
        display: block;
        font-weight: bold;
        font-size: 1em;
    }

    div.stButton > button:hover {
        background-color: #ffccd2;
        color: black;
    }

    /* Center text */
    .center-text {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1>FASHION CLASSIFIER & PRICE FINDER</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a clothing item image to predict its category and find pricing insights powered by AI.</p>", unsafe_allow_html=True)
st.info("📝 **Note**: This model was trained on Fashion-MNIST dataset and works best with simple clothing items on plain backgrounds.")

st.subheader("Upload an Image")
uploaded_file= st.file_uploader("Choose an image",type=['jpg','png','jpeg'])



if uploaded_file:
    st.image(uploaded_file , caption= "Uploaded Image", use_container_width=False)

    #predict
    # --- Prediction Trigger ---
    if st.button("Predict & Get Price Info"):
         with st.spinner("Analyzing your image... please wait ⏳"):
            try:
                # Load the trained model
                model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved', 'cnn_model.h5')
                model = tf.keras.models.load_model(model_path)
                class_names = get_class_names()
                
                # Preprocess uploaded image EXACTLY like the working test script
                img = Image.open(uploaded_file)
                
                # Step 1: Convert to grayscale
                img_gray = img.convert('L')
                
                # Step 2: Resize to 28x28
                img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Step 3: Convert to numpy array and normalize (EXACTLY like test script)
                img_array = np.array(img_resized).astype('float32') / 255.0
                
                # Step 4: Reshape for CNN input (EXACTLY like test script)
                img_array = img_array.reshape(1, 28, 28, 1)
                
                # Show original and preprocessed images
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_gray, caption="Original (Grayscale)", width=200)
                with col2:
                    st.image(img_array.reshape(28, 28), caption="Preprocessed (28x28)", width=200, clamp=True)
                
                st.info("📝 **Preprocessing Applied:** Converted to grayscale, resized to 28x28, normalized to [0,1] - same as training data")

                # Predict
                pred = model.predict(img_array, verbose=0)
                predicted_index = np.argmax(pred)
                predicted_label = class_names[predicted_index]
                confidence = np.max(pred)
                
                # Show prediction with confidence
                if confidence > 0.5:
                    st.success(f"🎯 Predicted Category: **{predicted_label}**")
                elif confidence > 0.3:
                    st.warning(f"🤔 Likely Category: **{predicted_label}** (Low confidence)")
                else:
                    st.error(f"❓ Uncertain: **{predicted_label}** (Very low confidence)")
                
                st.info(f"Confidence: {confidence:.2%}")
                
                # Show top 3 predictions
                top_3_indices = np.argsort(pred[0])[-3:][::-1]
                st.write("**Top 3 Predictions:**")
                for i, idx in enumerate(top_3_indices):
                    confidence_color = "🟢" if pred[0][idx] > 0.5 else "🟡" if pred[0][idx] > 0.3 else "🔴"
                    st.write(f"{confidence_color} {i+1}. **{class_names[idx]}**: {pred[0][idx]:.2%}")


                # --- Gemini API call ---
                if confidence > 0.2:  # Lower threshold since Fashion-MNIST is challenging
                    try:
                        api = FashionPriceAPI()
                        price_info = api.get_price_and_stores(predicted_label, "Kenya")
                        st.markdown("### 💰 Price Information")
                        st.info(price_info)
                    except Exception as api_error:
                        st.warning(f"Could not get price info: {api_error}")
                else:
                    st.warning("⚠️ Prediction confidence is very low. This model works best with simple clothing items on plain backgrounds, similar to Fashion-MNIST dataset.")
                    st.info("💡 **Tips for better predictions:**\n\n- Use images of single clothing items\n- Plain/simple backgrounds work best\n- Center the clothing item in the image\n- Avoid complex patterns or multiple items")


            except Exception as e:
                 st.error(f"Something went wrong: {e}")
                 st.text(str(e))
    
    


 
   

       