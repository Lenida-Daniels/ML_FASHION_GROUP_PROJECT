#!/usr/bin/env python3
"""
Generate test images from Fashion-MNIST CSV data for testing the Streamlit app
"""

import pandas as pd
import numpy as np
from PIL import Image
import os

def generate_test_images():
    """Generate sample test images from Fashion-MNIST CSV"""
    
    # Load test data
    test_df = pd.read_csv('data/raw/fashion-mnist_test.csv')
    
    # Class names
    class_names = ['T-shirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_boot']
    
    # Create test_images directory
    os.makedirs('test_images', exist_ok=True)
    
    print("🖼️ Generating test images from Fashion-MNIST data...")
    
    # Generate 5 samples from each class
    for class_idx in range(10):
        class_data = test_df[test_df['label'] == class_idx]
        class_name = class_names[class_idx]
        
        # Take first 5 samples of this class
        samples = class_data.head(5)
        
        for i, (_, row) in enumerate(samples.iterrows()):
            # Extract pixel data (skip label column)
            pixels = row.iloc[1:].values
            
            # Reshape to 28x28 image
            img_array = pixels.reshape(28, 28).astype(np.uint8)
            
            # Create PIL image
            img = Image.fromarray(img_array, mode='L')
            
            # Save image
            filename = f"{class_name}_{i+1}.png"
            filepath = os.path.join('test_images', filename)
            img.save(filepath)
            
            print(f"✅ Saved: {filename}")
    
    print(f"\n🎉 Generated 50 test images in 'test_images/' directory")
    print("\n📋 Test Images Created:")
    print("=" * 40)
    
    for class_idx, class_name in enumerate(class_names):
        print(f"{class_idx}: {class_name.replace('_', '/')} - 5 samples")
    
    print("\n🚀 Now you can test these images in your Streamlit app!")
    print("   Run: streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    generate_test_images()