#!/usr/bin/env python3
"""
Ultra-fast training script for deforestation detection model
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

print("="*60)
print("DEFORESTATION DETECTION MODEL - ULTRA FAST TRAINING")
print("="*60)

# Configuration
IMAGE_SIZE = (64, 64)  # Very small for speed
BATCH_SIZE = 16
EPOCHS = 5
SAMPLE_SIZE = 500  # Small sample

print(f"\n📋 Configuration:")
print(f"  - Image size: {IMAGE_SIZE}")
print(f"  - Sample size: {SAMPLE_SIZE}")

# Paths
dataset_dir = Path("data/raw/Forest Segmented/Forest Segmented")
images_dir = dataset_dir / "images"
masks_dir = dataset_dir / "masks"
meta_file = dataset_dir / "meta_data.csv"

print("\n🔄 Loading images...")

# Load metadata
meta_df = pd.read_csv(meta_file)
if len(meta_df) > SAMPLE_SIZE:
    meta_df = meta_df.sample(n=SAMPLE_SIZE, random_state=42)

images = []
labels = []

for idx, row in meta_df.iterrows():
    try:
        img_path = images_dir / row['image']
        mask_path = masks_dir / row['mask']
        
        if not img_path.exists(): continue
        
        # Read image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        images.append(img)
        
        # Read mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        forest_coverage = np.sum(mask > 0) / mask.size
        label = 0 if forest_coverage > 0.5 else 1
        labels.append(label)
        
        if len(images) % 100 == 0:
            print(f"  Loaded {len(images)} images...")
            
    except Exception:
        continue

images = np.array(images, dtype=np.float32) / 255.0
labels = np.array(labels)

print(f"✅ Loaded {len(images)} images")

# Split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print("\n🏗️ Building model...")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(*IMAGE_SIZE, 3)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n🚀 Training...")
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

print("\n💾 Saving model...")
model_save_path = Path("data/models/deforestation_model.h5")
model_save_path.parent.mkdir(parents=True, exist_ok=True)
model.save(model_save_path)

print(f"✅ Model saved to {model_save_path}")
