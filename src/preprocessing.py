import numpy as np

def preprocess_for_mlp(X_train, X_test, y_train, y_test):
    #Preprocess data for MLP
    #Normalize pixel Values
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    return X_train, X_test, y_train, y_test


def preprocess_for_cnn(X_train, X_test, y_train, y_test):
    #prepare data for CNN
    #Reshape for CNN 
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    return X_train, X_test, y_train, y_test