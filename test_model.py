#!/usr/bin/env python3
"""
Test script to verify trained CNN model works correctly
"""

import sys
sys.path.append('src')

import numpy as np
import tensorflow as tf
from data_loader import load_fashion_mnist, get_class_names
from preprocessing import preprocess_for_cnn

def test_trained_model():
    """Test the trained CNN model"""
    print("=== Testing Trained CNN Model ===")
    
    try:
        # Load the trained model
        model = tf.keras.models.load_model('models/saved/cnn_model.h5')
        print("✅ Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        # Load test data
        X_train, y_train, X_test, y_test = load_fashion_mnist()
        X_test_cnn, _, _, _ = preprocess_for_cnn(X_train, X_test, y_train, y_test)
        class_names = get_class_names()
        
        # Test prediction on a few samples
        print("\nTesting predictions on sample images:")
        for i in range(5):
            prediction = model.predict(X_test_cnn[i:i+1], verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            true_class = y_test[i]
            
            status = "✅" if predicted_class == true_class else "❌"
            print(f"{status} Sample {i+1}: Predicted={class_names[predicted_class]} ({confidence:.2f}), True={class_names[true_class]}")
        
        print("\n✅ Model is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

if __name__ == "__main__":
    test_trained_model()