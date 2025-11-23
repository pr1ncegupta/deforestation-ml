# Satellite Data for Deforestation Monitoring

A comprehensive machine learning project for monitoring and detecting deforestation using satellite imagery data from Kaggle, with an interactive Streamlit frontend for visualization and analysis.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Step-by-Step Setup Guide](#step-by-step-setup-guide)
- [Project Structure](#project-structure)
- [Dataset Information](#dataset-information)
- [Model Development](#model-development)
- [Running the Application](#running-the-application)
- [Deployment](#deployment)
- [Contributing](#contributing)

---

## 🌍 Project Overview

This project leverages satellite imagery and machine learning to detect and monitor deforestation patterns. The system analyzes satellite data to identify areas of forest loss, providing valuable insights for environmental conservation efforts.

**Key Objectives:**
- Detect deforestation patterns from satellite imagery
- Visualize temporal changes in forest cover
- Provide interactive analytics dashboard
- Generate actionable insights for conservation

---

## ✨ Features

- **Real-time Deforestation Detection**: ML-powered analysis of satellite imagery
- **Interactive Dashboard**: Streamlit-based UI for data exploration
- **Temporal Analysis**: Track deforestation trends over time
- **Geospatial Visualization**: Interactive maps showing affected areas
- **Statistical Reports**: Comprehensive analytics and metrics
- **Model Performance Metrics**: Accuracy, precision, recall, and F1-score tracking

---

## 🛠 Technology Stack

### Frontend
- **Streamlit**: Interactive web application framework
- **Plotly**: Interactive visualizations and charts
- **Folium**: Geospatial mapping
- **Matplotlib/Seaborn**: Statistical plots

### Backend & ML
- **Python 3.8+**: Core programming language
- **TensorFlow/Keras**: Deep learning framework
- **PyTorch**: Alternative deep learning framework (optional)
- **Scikit-learn**: Machine learning utilities
- **OpenCV**: Image processing
- **NumPy/Pandas**: Data manipulation

### Data Processing
- **Rasterio**: Geospatial raster data processing
- **GDAL**: Geospatial data abstraction library
- **Pillow**: Image processing

### Development Tools
- **Jupyter Notebook**: Exploratory data analysis
- **Git**: Version control
- **Docker**: Containerization (optional)

---

## 📦 Prerequisites

Before starting, ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package manager)
- Git
- Kaggle account and API credentials
- 8GB+ RAM recommended
- GPU (optional, but recommended for model training)

---

## 🚀 Step-by-Step Setup Guide

### **Step 1: Environment Setup**

#### 1.1 Create Project Directory
```bash
mkdir satellite-deforestation-monitoring
cd satellite-deforestation-monitoring
```

#### 1.2 Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### 1.3 Upgrade pip
```bash
pip install --upgrade pip
```

---

### **Step 2: Install Dependencies**

#### 2.1 Create requirements.txt
Create a file named `requirements.txt` with the following content:

```text
# Core Framework
streamlit==1.28.0

# Data Processing
pandas==2.1.0
numpy==1.24.3
pillow==10.0.0

# Machine Learning
tensorflow==2.13.0
scikit-learn==1.3.0
opencv-python==4.8.0.76

# Geospatial Libraries
rasterio==1.3.8
geopandas==0.13.2
shapely==2.0.1
pyproj==3.6.0

# Visualization
plotly==5.16.1
matplotlib==3.7.2
seaborn==0.12.2
folium==0.14.0
streamlit-folium==0.13.0

# Utilities
kaggle==1.5.16
python-dotenv==1.0.0
tqdm==4.66.1

# Optional: Deep Learning Enhancements
# torch==2.0.1
# torchvision==0.15.2
```

#### 2.2 Install Dependencies
```bash
pip install -r requirements.txt
```

---

### **Step 3: Kaggle API Setup**

#### 3.1 Get Kaggle API Credentials
1. Go to [kaggle.com](https://www.kaggle.com)
2. Sign in to your account
3. Click on your profile picture → "Settings"
4. Scroll to "API" section
5. Click "Create New API Token"
6. Download `kaggle.json` file

#### 3.2 Configure Kaggle API
```bash
# On macOS/Linux:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# On Windows:
# Create folder: C:\Users\<YourUsername>\.kaggle
# Move kaggle.json to that folder
```

#### 3.3 Verify Kaggle Setup
```bash
kaggle datasets list
```

---

### **Step 4: Download Dataset from Kaggle**

#### 4.1 Search for Deforestation Datasets
Recommended datasets:
- `"amazon-rainforest-satellite-images"`
- `"deforestation-in-amazon"`
- `"planet-understanding-the-amazon-from-space"`

#### 4.2 Download Dataset
```bash
# Example: Download Amazon deforestation dataset
kaggle datasets download -d mbogernetto/brazilian-amazon-rainforest-degradation

# Unzip the dataset
unzip brazilian-amazon-rainforest-degradation.zip -d data/raw/

# Or use Python script (create download_data.py):
```

Create `scripts/download_data.py`:
```python
import kaggle
import os
from zipfile import ZipFile

def download_dataset():
    # Download dataset
    dataset_name = "mbogernetto/brazilian-amazon-rainforest-degradation"
    download_path = "data/raw"
    
    os.makedirs(download_path, exist_ok=True)
    
    print(f"Downloading dataset: {dataset_name}")
    kaggle.api.dataset_download_files(
        dataset_name,
        path=download_path,
        unzip=True
    )
    print("Download complete!")

if __name__ == "__main__":
    download_dataset()
```

Run the script:
```bash
python scripts/download_data.py
```

---

### **Step 5: Project Structure Setup**

#### 5.1 Create Directory Structure
```bash
mkdir -p data/{raw,processed,models}
mkdir -p notebooks
mkdir -p src/{data,models,visualization,utils}
mkdir -p scripts
mkdir -p assets/{images,maps}
mkdir -p tests
touch src/__init__.py
```

#### 5.2 Final Project Structure
```
satellite-deforestation-monitoring/
│
├── data/
│   ├── raw/                    # Raw Kaggle datasets
│   ├── processed/              # Processed datasets
│   └── models/                 # Trained model files
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Data preprocessing
│   └── 03_model_training.ipynb # Model development
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Data loading utilities
│   │   └── preprocessor.py    # Data preprocessing
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_model.py       # CNN architecture
│   │   ├── trainer.py         # Model training
│   │   └── predictor.py       # Inference
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── maps.py            # Geospatial visualizations
│   │   └── charts.py          # Statistical charts
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py         # Utility functions
│
├── scripts/
│   ├── download_data.py       # Kaggle data download
│   └── train_model.py         # Model training script
│
├── assets/
│   ├── images/                # Static images
│   └── maps/                  # Generated maps
│
├── tests/                     # Unit tests
│
├── app.py                     # Main Streamlit application
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
├── .gitignore                # Git ignore file
└── README.md                 # This file
```

---

### **Step 6: Configuration Setup**

#### 6.1 Create config.py
```python
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Model parameters
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Model architecture
MODEL_NAME = "deforestation_cnn"
NUM_CLASSES = 2  # Deforested vs Non-deforested

# Visualization
MAP_CENTER = [-3.4653, -62.2159]  # Amazon coordinates
MAP_ZOOM = 6

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
```

#### 6.2 Create .env file

> **⚠️ SECURITY WARNING**: Never commit your `.env` file to Git! It contains sensitive API credentials.

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and fill in your actual Kaggle credentials
# You can get these from https://www.kaggle.com/settings
```

The `.env` file should contain:
- Your Kaggle username and API key (from kaggle.json)
- Model configuration paths
- Any other sensitive configuration

See `.env.example` for the complete template and detailed instructions.

#### 6.3 Create .gitignore
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
*.h5
*.pkl
*.pth

# Jupyter
.ipynb_checkpoints

# Environment
.env

# IDE
.vscode/
.idea/

# OS
.DS_Store
```

---

### **Step 7: Data Processing Pipeline**

#### 7.1 Create Data Loader (src/data/loader.py)
```python
import os
import numpy as np
from PIL import Image
from pathlib import Path
import rasterio
from typing import Tuple, List

class SatelliteDataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load satellite image"""
        with rasterio.open(image_path) as src:
            image = src.read()
            # Transpose to (H, W, C)
            image = np.transpose(image, (1, 2, 0))
        return image
    
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load entire dataset"""
        images = []
        labels = []
        
        # Implement your loading logic based on dataset structure
        # This is a template
        
        return np.array(images), np.array(labels)
```

#### 7.2 Create Preprocessor (src/data/preprocessor.py)
```python
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

class ImagePreprocessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size"""
        return cv2.resize(image, self.target_size)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize pixel values to [0, 1]"""
        return image.astype(np.float32) / 255.0
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation"""
        # Random flip
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
        
        # Random rotation
        angle = np.random.randint(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        
        return image
    
    def preprocess_dataset(self, images, labels, test_size=0.2):
        """Preprocess entire dataset"""
        processed_images = []
        
        for img in images:
            img = self.resize_image(img)
            img = self.normalize_image(img)
            processed_images.append(img)
        
        processed_images = np.array(processed_images)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            processed_images, labels, 
            test_size=test_size, 
            random_state=42
        )
        
        return X_train, X_test, y_train, y_test
```

---

### **Step 8: Model Development**

#### 8.1 Create CNN Model (src/models/cnn_model.py)
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DeforestationCNN:
    def __init__(self, input_shape=(256, 256, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        """Build CNN architecture"""
        model = keras.Sequential([
            # Convolutional Block 1
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Convolutional Block 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Convolutional Block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Convolutional Block 4
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile model"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_model(self):
        """Return compiled model"""
        return self.model
```

#### 8.2 Create Model Trainer (src/models/trainer.py)
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

class ModelTrainer:
    def __init__(self, model, model_save_path):
        self.model = model
        self.model_save_path = model_save_path
        self.history = None
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32):
        """Train the model"""
        
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                self.model_save_path,
                save_best_only=True,
                monitor='val_accuracy'
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        results = self.model.evaluate(X_test, y_test)
        return results
```

---

### **Step 9: Streamlit Application Development**

#### 9.1 Create Main App (app.py)
```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import folium
from streamlit_folium import folium_static
import tensorflow as tf

# Page configuration
st.set_page_config(
    page_title="Deforestation Monitoring System",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTitle {
        color: #2ecc71;
        font-size: 3rem !important;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🌳 Satellite Data for Deforestation Monitoring")
st.markdown("### AI-Powered Forest Conservation Analytics")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x150/2ecc71/ffffff?text=Forest+Monitor", 
             use_column_width=True)
    st.header("Navigation")
    page = st.radio("Go to", [
        "🏠 Dashboard",
        "📊 Data Analysis",
        "🤖 Model Prediction",
        "🗺️ Geospatial View",
        "📈 Statistics"
    ])
    
    st.markdown("---")
    st.info("**About**: This system uses satellite imagery and machine learning to detect deforestation patterns.")

# Dashboard Page
if page == "🏠 Dashboard":
    st.header("Dashboard Overview")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Area Monitored",
            value="15,420 km²",
            delta="↑ 5.2%"
        )
    
    with col2:
        st.metric(
            label="Deforestation Detected",
            value="342 km²",
            delta="↓ 12.3%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Model Accuracy",
            value="94.7%",
            delta="↑ 2.1%"
        )
    
    with col4:
        st.metric(
            label="Alerts Generated",
            value="23",
            delta="↑ 8"
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Deforestation Trend (2020-2024)")
        # Sample data
        years = [2020, 2021, 2022, 2023, 2024]
        deforestation = [450, 420, 380, 350, 342]
        
        fig = px.line(
            x=years, y=deforestation,
            labels={'x': 'Year', 'y': 'Area (km²)'},
            title="Annual Deforestation Rate"
        )
        fig.update_traces(line_color='#e74c3c', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Forest Cover Distribution")
        labels = ['Dense Forest', 'Moderate Forest', 'Sparse Forest', 'Deforested']
        values = [45, 30, 15, 10]
        
        fig = px.pie(
            values=values, names=labels,
            title="Current Forest Cover Status",
            color_discrete_sequence=px.colors.sequential.Greens_r
        )
        st.plotly_chart(fig, use_container_width=True)

# Data Analysis Page
elif page == "📊 Data Analysis":
    st.header("Data Analysis")
    
    st.subheader("Upload Satellite Image for Analysis")
    uploaded_file = st.file_uploader(
        "Choose a satellite image...", 
        type=['jpg', 'png', 'tif']
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Original Image", 
                    use_column_width=True)
        
        with col2:
            st.success("✅ Image loaded successfully!")
            st.write("**Image Properties:**")
            st.write(f"- Format: {uploaded_file.type}")
            st.write(f"- Size: {uploaded_file.size / 1024:.2f} KB")

# Model Prediction Page
elif page == "🤖 Model Prediction":
    st.header("AI Model Prediction")
    
    st.info("Upload a satellite image to get deforestation prediction")
    
    uploaded_file = st.file_uploader(
        "Choose an image for prediction", 
        type=['jpg', 'png']
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Input Image", use_column_width=True)
        
        with col2:
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing..."):
                    # Simulate prediction
                    import time
                    time.sleep(2)
                    
                    st.success("Analysis Complete!")
                    
                    # Display results
                    st.markdown("### Prediction Results")
                    st.progress(0.87)
                    st.write("**Deforestation Probability:** 87%")
                    st.write("**Classification:** High Risk")
                    
                    st.warning("⚠️ Significant deforestation detected in this area!")

# Geospatial View Page
elif page == "🗺️ Geospatial View":
    st.header("Geospatial Visualization")
    
    # Create map
    m = folium.Map(
        location=[-3.4653, -62.2159],  # Amazon coordinates
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Add markers (sample data)
    locations = [
        [-3.4653, -62.2159, "Alert Zone 1", "High"],
        [-3.1190, -60.0217, "Alert Zone 2", "Medium"],
        [-2.5297, -61.9628, "Alert Zone 3", "Low"]
    ]
    
    for lat, lon, name, severity in locations:
        color = 'red' if severity == 'High' else 'orange' if severity == 'Medium' else 'green'
        folium.Marker(
            [lat, lon],
            popup=f"{name}<br>Severity: {severity}",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)
    
    folium_static(m, width=1200, height=600)

# Statistics Page
elif page == "📈 Statistics":
    st.header("Detailed Statistics")
    
    # Sample data
    data = {
        'Region': ['North', 'South', 'East', 'West', 'Central'],
        'Forest Cover (%)': [78, 65, 82, 71, 69],
        'Deforestation Rate (%)': [2.3, 4.1, 1.8, 3.2, 3.5],
        'Alert Count': [5, 12, 3, 8, 7]
    }
    df = pd.DataFrame(data)
    
    st.dataframe(df, use_container_width=True)
    
    # Bar chart
    fig = px.bar(
        df, x='Region', y='Forest Cover (%)',
        title="Forest Cover by Region",
        color='Forest Cover (%)',
        color_continuous_scale='Greens'
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d;'>"
    "🌍 Satellite Deforestation Monitoring System | "
    "Powered by AI & Streamlit | © 2024"
    "</div>",
    unsafe_allow_html=True
)
```

---

### **Step 10: Training Script**

#### 10.1 Create Training Script (scripts/train_model.py)
```python
import sys
sys.path.append('..')

from src.models.cnn_model import DeforestationCNN
from src.models.trainer import ModelTrainer
from src.data.preprocessor import ImagePreprocessor
import config

def main():
    print("Starting model training...")
    
    # Initialize model
    cnn = DeforestationCNN(
        input_shape=(*config.IMAGE_SIZE, 3),
        num_classes=config.NUM_CLASSES
    )
    cnn.compile_model(learning_rate=config.LEARNING_RATE)
    
    # Load and preprocess data
    # (Add your data loading logic here)
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=cnn.get_model(),
        model_save_path=str(config.MODELS_DIR / f"{config.MODEL_NAME}.h5")
    )
    
    # Train model
    # history = trainer.train(X_train, y_train, X_val, y_val, 
    #                        epochs=config.EPOCHS, 
    #                        batch_size=config.BATCH_SIZE)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
```

---

### **Step 11: Running the Application**

#### 11.1 Start Streamlit App
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

#### 11.2 Train the Model
```bash
python scripts/train_model.py
```

---

### **Step 12: Testing**

#### 12.1 Create Test Files
Create `tests/test_model.py`:
```python
import unittest
import numpy as np
from src.models.cnn_model import DeforestationCNN

class TestDeforestationCNN(unittest.TestCase):
    def setUp(self):
        self.model = DeforestationCNN()
    
    def test_model_creation(self):
        self.assertIsNotNone(self.model.get_model())
    
    def test_model_input_shape(self):
        model = self.model.get_model()
        self.assertEqual(model.input_shape, (None, 256, 256, 3))

if __name__ == '__main__':
    unittest.main()
```

#### 12.2 Run Tests
```bash
python -m pytest tests/
```

---

### **Step 13: Deployment Options**

#### 13.1 Streamlit Cloud Deployment
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

#### 13.2 Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t deforestation-monitor .
docker run -p 8501:8501 deforestation-monitor
```

---

## 📊 Dataset Information

### Recommended Kaggle Datasets:

1. **Planet: Understanding the Amazon from Space**
   - Dataset: `planet-understanding-the-amazon-from-space`
   - Size: ~40GB
   - Images: Satellite imagery with labels

2. **Brazilian Amazon Rainforest Degradation**
   - Dataset: `mbogernetto/brazilian-amazon-rainforest-degradation`
   - Size: ~5GB
   - Time series data

3. **Amazon Rainforest Satellite Images**
   - Dataset: `nikitarom/amazon-rainforest-satellite-images`
   - Size: ~2GB
   - Labeled images

---

## 🎯 Model Development Workflow

1. **Exploratory Data Analysis (EDA)**
   - Analyze dataset structure
   - Visualize sample images
   - Check class distribution

2. **Data Preprocessing**
   - Resize images
   - Normalize pixel values
   - Apply augmentation

3. **Model Training**
   - Build CNN architecture
   - Train with validation
   - Monitor metrics

4. **Model Evaluation**
   - Test on holdout set
   - Generate confusion matrix
   - Calculate metrics

5. **Deployment**
   - Save trained model
   - Integrate with Streamlit
   - Deploy to cloud

---

## 🔧 Troubleshooting

### Common Issues:

**Issue**: Kaggle API not working
```bash
# Solution: Check credentials
cat ~/.kaggle/kaggle.json
```

**Issue**: Out of memory during training
```python
# Solution: Reduce batch size in config.py
BATCH_SIZE = 16  # Instead of 32
```

**Issue**: Streamlit not starting
```bash
# Solution: Check port availability
lsof -i :8501
# Kill process if needed
kill -9 <PID>
```

---

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Rasterio Documentation](https://rasterio.readthedocs.io)

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License.

---

## 👥 Authors

- Your Name - Initial work

---

## 🙏 Acknowledgments

- Kaggle for providing datasets
- Streamlit for the amazing framework
- TensorFlow team for ML tools

---

**Happy Coding! 🚀🌳**
