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

# Kaggle dataset
KAGGLE_DATASET = "mbogernetto/brazilian-amazon-rainforest-degradation"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
