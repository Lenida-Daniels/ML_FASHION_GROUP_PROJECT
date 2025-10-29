#!/usr/bin/env python3
"""Test script for data preprocessing functions"""

import sys
import os
sys.path.append('src')

from data_loader import load_fashion_mnist, get_class_names
from preprocessing import preprocess_for_mlp, preprocess_for_cnn

def test_data_loading():
    """Test if data loading works"""
    print("Testing data loading...")
    try:
        X_train, y_train, X_test, y_test = load_fashion_mnist()
        print(f"✅ Data loaded successfully!")
        print(f"   Training data shape: {X_train.shape}")
        print(f"   Test data shape: {X_test.shape}")
        print(f"   Training labels shape: {y_train.shape}")
        print(f"   Test labels shape: {y_test.shape}")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None, None, None, None

def test_preprocessing(X_train, y_train, X_test, y_test):
    """Test preprocessing functions"""
    if X_train is None:
        print("❌ Skipping preprocessing tests - no data")
        return
    
    print("\nTesting MLP preprocessing...")
    try:
        X_train_mlp, X_test_mlp, y_train_mlp, y_test_mlp = preprocess_for_mlp(X_train, X_test, y_train, y_test)
        print(f"✅ MLP preprocessing successful!")
        print(f"   MLP data range: {X_train_mlp.min():.2f} to {X_train_mlp.max():.2f}")
    except Exception as e:
        print(f"❌ MLP preprocessing failed: {e}")
    
    print("\nTesting CNN preprocessing...")
    try:
        X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = preprocess_for_cnn(X_train, X_test, y_train, y_test)
        print(f"✅ CNN preprocessing successful!")
        print(f"   CNN data shape: {X_train_cnn.shape}")
        print(f"   CNN data range: {X_train_cnn.min():.2f} to {X_train_cnn.max():.2f}")
    except Exception as e:
        print(f"❌ CNN preprocessing failed: {e}")

def test_class_names():
    """Test class names function"""
    print("\nTesting class names...")
    try:
        classes = get_class_names()
        print(f"✅ Class names loaded: {len(classes)} classes")
        for i, name in enumerate(classes):
            print(f"   {i}: {name}")
    except Exception as e:
        print(f"❌ Class names failed: {e}")

if __name__ == "__main__":
    print("=== Testing Data Preprocessing Functions ===")
    
    # Test data loading
    X_train, y_train, X_test, y_test = test_data_loading()
    
    # Test preprocessing
    test_preprocessing(X_train, y_train, X_test, y_test)
    
    # Test class names
    test_class_names()
    
    print("\n=== Test Complete ===")