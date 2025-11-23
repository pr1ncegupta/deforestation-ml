#!/usr/bin/env python3
import sys
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

print("Creating model structure...")
model = models.Sequential([
    layers.Input(shape=(256, 256, 3)),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Saving model...")
model_save_path = Path("data/models/deforestation_model.h5")
model_save_path.parent.mkdir(parents=True, exist_ok=True)
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
