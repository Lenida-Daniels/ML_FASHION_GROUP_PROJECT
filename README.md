# Fashion Classification with Neural Networks

## Project Overview
This project implements and compares neural network approaches for fashion item classification using the Fashion-MNIST dataset. Users can upload clothing images to get instant classification results along with price estimates and shopping recommendations powered by Google's Gemini AI.

## Features
- **Image Classification**: Upload clothing images and get accurate category predictions
- **Model Comparison**: Compare Multi-Layer Perceptron (MLP) vs Convolutional Neural Network (CNN) performance
- **Price Intelligence**: Get real-time price estimates for classified clothing items
- **Shopping Assistant**: Receive store recommendations and purchase suggestions
- **Interactive Web App**: User-friendly Streamlit interface for easy interaction

## Dataset
- **Fashion-MNIST**: 70,000 grayscale images (28x28 pixels)
- **Categories**: 10 clothing types (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Split**: 60,000 training + 10,000 test images

## Technical Implementation

### Core Models
1. **MLP Baseline**: Simple fully connected neural network
2. **CNN Model**: Convolutional neural network optimized for image data

### AI Integration
- **Gemini API**: Provides price estimates and store recommendations
- **Real-time Processing**: Instant classification and price lookup

### Performance Analysis
- Model accuracy comparison and evaluation
- Visualization of incorrect predictions
- Analysis of CNN advantages over MLP for image data

## Project Structure
```
fashion-classifier/
├── src/
│   ├── data_loader.py      # Fashion-MNIST data loading
│   └── preprocessing.py    # Data preprocessing for MLP/CNN
├── models/
│   ├── mlp_model.py       # Multi-Layer Perceptron
│   └── cnn_model.py       # Convolutional Neural Network
├── api/
│   └── gemini_integration.py  # Price & store lookup
├── app/
│   └── streamlit_app.py   # Web interface
├── test_*.py              # Testing scripts
└── requirements.txt       # Dependencies
```

## Installation
```bash
git clone https://github.com/Lenida-Daniels/ML_FASHION_GROUP_PROJECT.git
cd ML_FASHION_GROUP_PROJECT
pip install -r requirements.txt
```

## Usage
1. **Set up Gemini API**:
   ```bash
   export GEMINI_API_KEY='your-api-key'
   ```

2. **Test components**:
   ```bash
   python test_preprocessing.py
   python test_gemini.py
   ```

3. **Run web app**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Real-World Applications
- **E-commerce**: Automatic product categorization
- **Inventory Management**: Smart clothing classification
- **Fashion Apps**: Personal wardrobe organization
- **Retail**: Price comparison and shopping assistance

## Team Contributions
- **Data Preprocessing & API Integration**: Vincent
- **MLP Model Development**: [Team Member]
- **CNN Model Development**: [Team Member]
- **Web Interface**: [Team Member]
- **Model Analysis & Evaluation**: [Team Member]

## Technologies Used
- **Machine Learning**: TensorFlow/Keras, scikit-learn
- **AI Integration**: Google Gemini API
- **Web Framework**: Streamlit
- **Data Processing**: NumPy
- **Visualization**: Matplotlib