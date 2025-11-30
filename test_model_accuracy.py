#!/usr/bin/env python3
"""
Test the saved model accuracy on Fashion-MNIST test data
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import tensorflow as tf
from data_loader import get_class_names
from preprocessing import preprocess_for_cnn

def test_model_accuracy():
    """Test model on actual Fashion-MNIST test data"""
    print("🧪 Testing Model Accuracy on Fashion-MNIST Test Data")
    print("=" * 60)
    
    # Load model
    try:
        model = tf.keras.models.load_model('models/saved/cnn_model.h5')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Load test data from CSV
    try:
        test_df = pd.read_csv('data/raw/fashion-mnist_test.csv')
        print(f"✅ Test data loaded: {test_df.shape}")
        
        # Extract features and labels
        X_test = test_df.drop('label', axis=1).values
        y_test = test_df['label'].values
        
        print(f"   Test images: {X_test.shape}")
        print(f"   Test labels: {y_test.shape}")
        
    except Exception as e:
        print(f"❌ Failed to load test data: {e}")
        return
    
    # Preprocess for CNN
    try:
        # Reshape to 28x28x1 and normalize
        X_test_cnn = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        print(f"✅ Data preprocessed: {X_test_cnn.shape}")
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return
    
    # Test on first 100 samples for speed
    n_samples = 100
    X_test_sample = X_test_cnn[:n_samples]
    y_test_sample = y_test[:n_samples]
    
    print(f"\n🔍 Testing on first {n_samples} samples...")
    
    # Make predictions
    try:
        predictions = model.predict(X_test_sample, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(pred_classes == y_test_sample)
        print(f"✅ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Show some examples
        class_names = get_class_names()
        print(f"\n📊 Sample Predictions:")
        print("-" * 50)
        
        correct = 0
        for i in range(min(10, n_samples)):
            true_label = y_test_sample[i]
            pred_label = pred_classes[i]
            confidence = np.max(predictions[i])
            
            status = "✅" if pred_label == true_label else "❌"
            print(f"{status} Sample {i+1}: True={class_names[true_label]}, Pred={class_names[pred_label]} ({confidence:.3f})")
            
            if pred_label == true_label:
                correct += 1
        
        print(f"\nFirst 10 samples accuracy: {correct}/10 = {correct/10*100:.1f}%")
        
        if accuracy < 0.5:
            print("\n⚠️  WARNING: Model accuracy is very low!")
            print("   This suggests the model wasn't trained properly or")
            print("   there's a preprocessing mismatch.")
            
            # Check if model was actually trained
            print(f"\n🔍 Model Summary:")
            model.summary()
            
        else:
            print(f"\n✅ Model appears to be working correctly!")
            
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return

if __name__ == "__main__":
    test_model_accuracy()