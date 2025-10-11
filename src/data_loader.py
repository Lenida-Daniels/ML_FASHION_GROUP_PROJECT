import numpy as np
from tensorflow.keras.datasets import fashion_mnist

def load_fashion_mnist():
    """Load Fashion-MNIST from Keras built-in dataset"""
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Flatten images for MLP (28x28 -> 784)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    return X_train, y_train, X_test, y_test


def get_class_names():
    #Fashion-MNIST class names
    return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']    
