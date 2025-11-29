import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from api.gemini_intergration import FashionPriceAPI
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
st.markdown("<p class='subtitle'>Upload your outfit to predict its category and find pricing insights powered by AI.</p>", unsafe_allow_html=True)

st.subheader("Upload an Image")
uploaded_file= st.file_uploader("Choose an image",type=['jpg','png','jpeg'])



if uploaded_file:
    st.image(uploaded_file , caption= "Uploaded Image", use_container_width=False)

    #predict
    # --- Prediction Trigger ---
    if st.button("Predict & Get Price Info"):
         with st.spinner("Analyzing your image... please wait ⏳"):
            try:
                # preprocess uploaded image
                def preprocess_single_image(img):
                       # Convert to grayscale
                        img = img.convert("L")
                        # Resize to 28x28
                        img = img.resize((28, 28))
                        # Convert to numpy array
                        img_array = np.array(img) / 255.0
                        # Reshape to (1, 28, 28, 1) for CNN input
                        img_array = img_array.reshape(1, 28, 28, 1)
                        return img_array
                

                model = load_model("cnn_model.h5")

                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                
                # Preprocess uploaded image
                img_array = preprocess_single_image(uploaded_file)

                # Predict
                pred = model.predict(img_array)
                predicted_index = np.argmax(pred)
                predicted_label = class_names[predicted_index]

                st.success(f"Predicted Category: {predicted_label}")


                # --- Gemini API call ---
                api = FashionPriceAPI()
                price_info = api.get_price_and_stores(predicted_label)
                st.info(price_info)


            except Exception as e:
                 st.error(f"Something went wrong: {e}")
                 st.text(traceback.format_exc())
    
    


 
   

       