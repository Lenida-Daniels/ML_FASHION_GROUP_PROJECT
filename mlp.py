import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import json 

def load_fashion_mnist():
    
    print("Loading Fashion-MNIST data...")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    return X_train, y_train, X_test, y_test

def preprocess_for_mlp(X_train, X_test, y_train, y_test):
    
    print("Preprocessing data (Normalization and Flattening)...")

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    X_train = X_train.reshape((-1, 784))
    X_test = X_test.reshape((-1, 784))

    return X_train, X_test, y_train, y_test

def build_mlp_model(input_shape=(784,), num_classes=10):
    
    print("Building MLP model architecture...")
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Dense(256, activation='relu', name='Dense_1'),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu', name='Dense_2'),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation='softmax', name='Output')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    model.summary()
    return model


def train_mlp_model(epochs=10, batch_size=64):
    
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_fashion_mnist()
    X_train, X_test, y_train, y_test = preprocess_for_mlp(
        X_train_raw, X_test_raw, y_train_raw, y_test_raw
    )

    model = build_mlp_model(input_shape=(X_train.shape[1],))
    
    print(f"\n--- Starting Training for {epochs} Epochs ---\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "fashion_mlp.h5")
    model.save(model_path)
    print(f"\nMLP model successfully trained and saved as {model_path}")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    history_path = os.path.join(results_dir, "mlp_training_history.json")
    
    history_dict = {k: [float(v_i) for v_i in v] for k, v in history.history.items()}
    
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=4)
    print(f"Training history saved as {history_path}")

    return model, history, X_test, y_test


def load_trained_model(model_path="models/fashion_mlp.h5"):
    
    return tf.keras.models.load_model(model_path)


if __name__ == "__main__":
    
    trained_model, training_history, X_test_data, y_test_data = train_mlp_model(epochs=5, batch_size=128)
    
    print("\n--- Evaluating Final Model ---")
    loss, accuracy = trained_model.evaluate(X_test_data, y_test_data, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    loaded_model = load_trained_model()
    print("\nSuccessfully loaded the model for deployment.")