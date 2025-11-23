#!/usr/bin/env python3
"""
Quick training script for deforestation detection model
Uses forest segmentation masks to create binary classifications
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import config

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

print("="*60)
print("DEFORESTATION DETECTION MODEL - QUICK TRAINING")
print("="*60)

# Configuration
IMAGE_SIZE = (128, 128)  # Smaller for faster training
BATCH_SIZE = 32
EPOCHS = 10  # Quick training
LEARNING_RATE = 0.001
SAMPLE_SIZE = 2000  # Use subset for quick training

print(f"\n📋 Training Configuration:")
print(f"  - Image size: {IMAGE_SIZE}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Sample size: {SAMPLE_SIZE} images")
print(f"  - Learning rate: {LEARNING_RATE}")

# Paths
dataset_dir = Path("data/raw/Forest Segmented/Forest Segmented")
images_dir = dataset_dir / "images"
masks_dir = dataset_dir / "masks"
meta_file = dataset_dir / "meta_data.csv"

print(f"\n📂 Loading dataset from: {dataset_dir}")

# Load metadata
meta_df = pd.read_csv(meta_file)
print(f"  ✅ Found {len(meta_df)} image-mask pairs")

# Sample for quick training
if len(meta_df) > SAMPLE_SIZE:
    meta_df = meta_df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"  ✅ Using {SAMPLE_SIZE} samples for quick training")

print("\n🔄 Loading and processing images...")

images = []
labels = []
processed_count = 0

for idx, row in meta_df.iterrows():
    try:
        # Load image
        img_path = images_dir / row['image']
        mask_path = masks_dir / row['mask']
        
        if not img_path.exists() or not mask_path.exists():
            continue
        
        # Read image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        images.append(img)
        
        # Read mask and calculate forest coverage
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        forest_coverage = np.sum(mask > 0) / mask.size
        
        # Label: 0 = Non-deforested (>50% forest), 1 = Deforested (<50% forest)
        label = 0 if forest_coverage > 0.5 else 1
        labels.append(label)
        
        processed_count += 1
        if processed_count % 200 == 0:
            print(f"  Processed {processed_count}/{SAMPLE_SIZE} images...")
            
    except Exception as e:
        continue

print(f"\n✅ Successfully loaded {len(images)} images")

# Convert to numpy arrays
images = np.array(images, dtype=np.float32) / 255.0  # Normalize
labels = np.array(labels)

print(f"\n📊 Label Distribution:")
unique, counts = np.unique(labels, return_counts=True)
for label, count in zip(unique, counts):
    label_name = "Deforested" if label == 1 else "Non-Deforested"
    print(f"  - {label_name}: {count} ({count/len(labels)*100:.1f}%)")

# Split dataset
print("\n🔪 Splitting dataset...")
X_train, X_temp, y_train, y_temp = train_test_split(
    images, labels, test_size=0.3, random_state=42, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"  - Training: {len(X_train)} images")
print(f"  - Validation: {len(X_val)} images")
print(f"  - Testing: {len(X_test)} images")

# Build model
print("\n🏗️ Building CNN model...")

model = models.Sequential([
    # Input layer
    layers.Input(shape=(*IMAGE_SIZE, 3)),
    
    # Conv Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Conv Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Conv Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Dense Layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  ✅ Model created with {model.count_params():,} parameters")

# Callbacks
model_save_path = Path("data/models/deforestation_model.h5")
model_save_path.parent.mkdir(parents=True, exist_ok=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(model_save_path),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# Train model
print(f"\n🚀 Starting training for {EPOCHS} epochs...")
print("="*60)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Evaluate on test set
print("\n📊 Evaluating on test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

print(f"\n📈 Final Results:")
print(f"  - Test Loss: {test_loss:.4f}")
print(f"  - Test Accuracy: {test_accuracy*100:.2f}%")

# Make sample predictions
print(f"\n🔮 Sample Predictions on Test Set:")
sample_preds = model.predict(X_test[:5], verbose=0)
for i, (pred, true_label) in enumerate(zip(sample_preds, y_test[:5])):
    pred_class = np.argmax(pred)
    confidence = pred[pred_class]
    pred_name = "Deforested" if pred_class == 1 else "Non-Deforested"
    true_name = "Deforested" if true_label == 1 else "Non-Deforested"
    match = "✓" if pred_class == true_label else "✗"
    print(f"  Sample {i+1}: Pred={pred_name} ({confidence:.2%}) | True={true_name} {match}")

print(f"\n💾 Model saved to: {model_save_path}")
print(f"\n🎉 You can now use AI predictions in the Streamlit app!")
print(f"   Go to: http://localhost:8501")
print(f"   Navigate to: 🤖 AI Prediction")
print(f"   Upload an image and get real predictions!")

print("\n" + "="*60)
