# Alphanumeric Character Recognition using CNNs

A comprehensive Deep Learning project that implements handwritten alphanumeric character recognition using Convolutional Neural Networks (CNNs). The system can recognize digits (0-9) and uppercase letters (A-Z) with high accuracy and includes a web-based drawing interface for real-time predictions.

Dataset link: https://www.kaggle.com/datasets/crawford/emnist

## 🎯 Project Overview

This project demonstrates the implementation of a CNN-based character recognition system trained on the EMNIST (Extended MNIST) dataset. The system achieves over 92% accuracy on validation data and includes both a Jupyter notebook for model development/training and a Flask web application for interactive testing.

## ✨ Features

- **36-class recognition**: Digits 0-9 and uppercase letters A-Z
- **High accuracy**: 95%+ training accuracy, 92%+ validation accuracy
- **Interactive web interface**: Draw characters and get real-time predictions
- **Confidence scoring**: Shows prediction confidence and top-3 alternatives
- **Model persistence**: Trained models saved for deployment
- **Comprehensive evaluation**: Detailed performance metrics and visualizations

## 🎨 Demo

![Web Application Demo](images/demo-screenshot.png)

*The web interface in action: Drawing the letter "G" with 100% confidence prediction and debug visualization showing the preprocessed image.*

## 🏗️ Architecture

### CNN Model Architecture
- **Input Layer**: 28×28 grayscale images
- **Conv2D Layer 1**: 32 filters, 3×3 kernel, ReLU activation
- **MaxPooling2D**: 2×2 pool size
- **Conv2D Layer 2**: 64 filters, 3×3 kernel, ReLU activation
- **MaxPooling2D**: 2×2 pool size
- **Conv2D Layer 3**: 64 filters, 3×3 kernel, ReLU activation
- **Flatten Layer**: Convert to 1D
- **Dense Layer**: 64 units, ReLU activation
- **Dropout**: 0.5 for regularization
- **Output Layer**: 36 units (softmax activation)

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Batch Size**: 16
- **Epochs**: 20
- **Validation Split**: 20%

## 📁 Project Structure

```
DSP_Project/
├── src/                       # Source code
│   ├── app.py                # Flask web application
│   ├── templates/            # Web interface templates
│   │   └── index.html       # Main drawing interface
│   └── saved_models/         # Trained model files
│       └── cnn_alphanumeric_model_20250601_062953.h5
├── notebooks/                 # Jupyter notebooks
│   └── neitbook.ipynb        # Model development and training
├── images/                    # Screenshots and demo images
│   ├── demo-screenshot.png   # Web app demo
│   └── image.png            # Additional images
├── docs/                      # Documentation (future use)
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker container configuration
├── docker-compose.yml         # Docker Compose orchestration
├── .dockerignore             # Docker ignore file
└── README.md                  # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

**OR**

- Docker and Docker Compose (recommended for easy deployment)

### Installation

#### Option 1: Local Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DSP_Project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web application**:
   ```bash
   cd src
   python app.py
   ```

4. **Access the interface**:
   Open your browser and navigate to `http://localhost:5000`

#### Option 2: Docker Deployment (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DSP_Project
   ```

2. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

3. **Access the interface**:
   Open your browser and navigate to `http://localhost:5000`

#### Option 3: Docker Manual Build

1. **Build the Docker image**:
   ```bash
   docker build -t character-recognition .
   ```

2. **Run the container**:
   ```bash
   docker run -p 5000:5000 character-recognition
   ```

### Training Your Own Model

1. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook notebooks/neitbook.ipynb
   ```

2. **Run all cells** to:
   - Load and preprocess the EMNIST dataset
   - Build the CNN architecture
   - Train the model
   - Evaluate performance
   - Save the trained model

## 🎮 Usage

### Web Interface
1. Visit `http://localhost:5000` after running the Flask app
2. Draw a character (digit or uppercase letter) in the canvas
3. Click "Predict" to get the model's prediction
4. View confidence scores and alternative predictions
5. Use "Clear Canvas" to start over

### Model Training
The Jupyter notebook (`notebooks/neitbook.ipynb`) contains comprehensive code for:
- Data loading and preprocessing
- Model architecture definition
- Training with validation
- Performance evaluation
- Model saving

## 📊 Performance Metrics

### Model Performance
- **Training Accuracy**: 95.26%
- **Validation Accuracy**: 92.04%
- **Training Loss**: 0.1159
- **Validation Loss**: 0.3681

### Class-wise Performance
The model tries to handle challenging cases like distinguishing between:
- '0' (zero) vs 'O' (letter O)
- '1' (one) vs 'I' (letter I)
- '5' vs 'S'
- '2' vs 'Z'

## 🔧 Technical Details

### Data Preprocessing
- **Normalization**: Pixel values scaled to [0, 1]
- **Orientation correction**: EMNIST images rotated and flipped
- **One-hot encoding**: Labels converted for categorical classification
- **Class filtering**: Only digits (0-9) and uppercase letters (A-Z)

### Web Application Features
- **Canvas drawing**: HTML5 canvas with touch/mouse support
- **Image preprocessing**: Automatic cropping, resizing, and normalization
- **Real-time prediction**: Instant results via AJAX
- **Debug visualization**: View preprocessed image

### Model Deployment
- **Flask backend**: Serves the trained model
- **Docker containerization**: Easy deployment with Docker and Docker Compose
- **Production ready**: Configured for production environments
- **Health checks**: Built-in container health monitoring
- **TensorFlow.js ready**: Model can be converted for client-side inference
- **API endpoints**: `/predict` for predictions, `/debug_image` for preprocessing visualization

## 📈 Results and Evaluation

The model demonstrates excellent performance across all character classes with:
- Robust handling of various handwriting styles
- High confidence in correct predictions
- Meaningful uncertainty quantification
- Good generalization to new handwriting samples

---
