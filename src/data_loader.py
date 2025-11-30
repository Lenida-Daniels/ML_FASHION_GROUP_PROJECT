import pandas as pd
import numpy as np
import os

def load_fashion_mnist():
    """Load Fashion-MNIST from local CSV files"""
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Paths to CSV files
    train_path = os.path.join(project_root, 'data', 'raw', 'fashion-mnist_train.csv')
    test_path = os.path.join(project_root, 'data', 'raw', 'fashion-mnist_test.csv')
    
    # Check if files exist
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"CSV files not found. Using Keras dataset instead.")
        # Fallback to Keras dataset
        from tensorflow.keras.datasets import fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        return X_train, y_train, X_test, y_test
    
    # Load from CSV files
    print(f"Loading data from local CSV files...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Separate features and labels
    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values
    
    return X_train, y_train, X_test, y_test


def get_class_names():
    #Fashion-MNIST class names
    return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']    
