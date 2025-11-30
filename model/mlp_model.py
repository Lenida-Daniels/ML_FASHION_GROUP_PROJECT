import sys
import os
sys.path.append('../src')

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from data_loader import load_fashion_mnist, get_class_names
from preprocessing import preprocess_for_mlp

def create_mlp_model():
    """Create MLP model architecture"""
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_mlp_model():
    """Train MLP model using preprocessing functions"""
    print("=== Training MLP Model ===")
    
    # Load data using your data_loader
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    
    # Preprocess using your preprocessing function
    X_train, X_test, y_train, y_test = preprocess_for_mlp(X_train, X_test, y_train, y_test)
    
    # Create model
    model = create_mlp_model()
    
    print("Model architecture:")
    model.summary()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=128,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nMLP Test Accuracy: {test_acc:.4f}")
    
    # Save model
    os.makedirs('../models/saved', exist_ok=True)
    model.save('../models/saved/mlp_model.h5')
    
    return model, history, test_acc

if __name__ == "__main__":
    model, history, accuracy = train_mlp_model()