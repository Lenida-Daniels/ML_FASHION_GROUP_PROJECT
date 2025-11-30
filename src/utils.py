"""
Utility functions that don't require data loading
"""

def get_class_names():
    """Fashion-MNIST class names - no data loading required"""
    return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocess_single_image_for_cnn(img):
    """Preprocess a single PIL image for CNN prediction"""
    import numpy as np
    
    # Convert to grayscale
    img = img.convert("L")
    # Resize to 28x28
    img = img.resize((28, 28))
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    # Reshape to (1, 28, 28, 1) for CNN input
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array