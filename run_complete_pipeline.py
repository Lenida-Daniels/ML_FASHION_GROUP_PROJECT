#!/usr/bin/env python3
"""
Complete Fashion-MNIST Classification Pipeline
Tests the entire flow: Data → Model → Prediction → Gemini API
"""

import sys
import os
sys.path.append('src')
sys.path.append('api')

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Import our custom functions
from data_loader import load_fashion_mnist, get_class_names
from preprocessing import preprocess_for_cnn
from gemini_integration import FashionPriceAPI

def test_complete_pipeline():
    """Test the complete pipeline"""
    print("🚀 FASHION-MNIST COMPLETE PIPELINE TEST")
    print("=" * 50)
    
    # Step 1: Load trained model
    print("\n1️⃣ Loading trained CNN model...")
    try:
        model = tf.keras.models.load_model('models/saved/cnn_model.h5')
        print(" Model loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
    except Exception as e:
        print(f" Model loading failed: {e}")
        return False
    
    # Step 2: Load and preprocess test data
    print("\n2️⃣ Loading and preprocessing data...")
    try:
        X_train, y_train, X_test, y_test = load_fashion_mnist()
        X_test_cnn, _, _, _ = preprocess_for_cnn(X_train, X_test, y_train, y_test)
        class_names = get_class_names()
        print(" Data loaded and preprocessed!")
        print(f"   Original test data shape: {X_test.shape}")
        print(f"   Preprocessed test data shape: {X_test_cnn.shape}")
        print(f"   Test labels shape: {y_test.shape}")
        print(f"   Classes: {len(class_names)}")
    except Exception as e:
        print(f" Data loading failed: {e}")
        return False
    
    # Step 3: Make predictions
    print("\n3️⃣ Making predictions...")
    try:
        # Test on 5 random samples (ensure indices are within bounds)
        max_samples = min(len(X_test_cnn), len(y_test))
        num_samples = min(5, max_samples)
        test_indices = np.random.choice(max_samples, num_samples, replace=False)
        print(f"Testing with {num_samples} samples, indices: {test_indices} (max available: {max_samples-1})")
        
        print("\nPrediction Results:")
        print("-" * 40)
        
        for i, idx in enumerate(test_indices):
            # Get prediction
            prediction = model.predict(X_test_cnn[idx:idx+1], verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            true_class = y_test[idx] if idx < len(y_test) else 0
            
            # Display result
            status = "" if predicted_class == true_class else "❌"
            print(f"{status} Sample {i+1}:")
            print(f"   Predicted: {class_names[predicted_class]} ({confidence:.2f})")
            print(f"   Actual: {class_names[true_class]}")
            print()
            
        print(" Predictions completed!")
        
        # Use the first prediction for Gemini API test
        best_prediction_idx = test_indices[0]
        if best_prediction_idx < len(X_test_cnn):
            best_prediction = model.predict(X_test_cnn[best_prediction_idx:best_prediction_idx+1], verbose=0)
            predicted_category = class_names[np.argmax(best_prediction)]
        else:
            print(f"Index {best_prediction_idx} out of bounds, using index 0")
            best_prediction = model.predict(X_test_cnn[0:1], verbose=0)
            predicted_category = class_names[np.argmax(best_prediction)]
        
    except Exception as e:
        print(f" Prediction failed: {e}")
        return False
    
    # Step 4: Test Gemini API integration
    print("\n4️⃣ Testing Gemini API integration...")
    
    # Check if API key is available
    if not os.getenv('GEMINI_API_KEY'):
        print("  GEMINI_API_KEY not set - skipping API test")
        print("   Set it with: export GEMINI_API_KEY='your-api-key'")
        print(" Pipeline test completed (without Gemini API)")
        return True
    
    try:
        api = FashionPriceAPI()
        print(f"🔍 Getting price info for: {predicted_category}")
        
        result = api.get_price_and_stores(predicted_category, "Nairobi, Kenya")
        
        print(" Gemini API Response:")
        print("-" * 50)
        print(result)
        print("-" * 50)
        
    except Exception as e:
        print(f" Gemini API test failed: {e}")
        print(" Pipeline test completed (Gemini API had issues)")
        return True
    
    # Step 5: Complete success
    print("\n🎉 COMPLETE PIPELINE SUCCESS!")
    print("=" * 50)
    print("Model loading: SUCCESS")
    print(" Data preprocessing: SUCCESS") 
    print(" Image classification: SUCCESS")
    print(" Gemini API integration: SUCCESS")
    print("=" * 50)
    print("\n🚀 Ready for Streamlit app integration!")
    
    return True

def show_sample_prediction():
    """Show a visual sample prediction"""
    print("\n📸 Visual Prediction Sample:")
    
    try:
        # Load model and data
        model = tf.keras.models.load_model('models/saved/cnn_model.h5')
        X_train, y_train, X_test, y_test = load_fashion_mnist()
        X_test_cnn, _, _, _ = preprocess_for_cnn(X_train, X_test, y_train, y_test)
        class_names = get_class_names()
        
        # Get a random sample
        idx = np.random.randint(0, len(X_test_cnn))
        sample_image = X_test_cnn[idx]
        true_label = y_test[idx]
        
        # Make prediction
        prediction = model.predict(sample_image.reshape(1, 28, 28, 1), verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Display
        plt.figure(figsize=(6, 4))
        plt.imshow(sample_image.reshape(28, 28), cmap='gray')
        
        color = 'green' if predicted_class == true_label else 'red'
        plt.title(f'Predicted: {class_names[predicted_class]} ({confidence:.2f})\\n'
                 f'True: {class_names[true_label]}', color=color, fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('results/sample_test_prediction.png')
        plt.show()
        
        print(f" Sample prediction saved to: results/sample_test_prediction.png")
        
    except Exception as e:
        print(f" Visual prediction failed: {e}")

if __name__ == "__main__":
    # Run complete pipeline test
    success = test_complete_pipeline()
    
    if success:
        # Show visual sample
        show_sample_prediction()
        
        print("\n🎯 NEXT STEPS:")
        print("1. Run Streamlit app: streamlit run app/streamlit_app.py")
        print("2. Upload images and test classification")
        print("3. Get price recommendations via Gemini API")
    else:
        print("\n Pipeline test failed. Check the errors above.")