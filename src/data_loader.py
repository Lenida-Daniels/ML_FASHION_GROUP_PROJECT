import pandas as pd
import numpy as np

def load_fashion_mnist():
    #Load Fashion-MNIST from CSV files
    train_df = pd.read_csv("fashion/fashion-mnist_train.csv")
    test_df = pd.read_csv("fashion/fashion-mnist_test.csv")

    # seperate features and labels
    X_train = train_df.drop("label", axis=1).values
    y_train = train_df["label"].values
    X_test = test-df.drop("label", axis=1).values
    y_test = test_df["label"].values

    return X_train, y_train, X_test, y_test


def get_class_names():
    #Fashion-MNIST class names
    return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']    
