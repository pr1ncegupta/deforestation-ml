#!/usr/bin/env python3
"""
Robust training script using Scikit-Learn (Random Forest)
This avoids TensorFlow compatibility issues on some Mac environments
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

print("="*60)
print("DEFORESTATION DETECTION - ROBUST TRAINING (RF)")
print("="*60)

# Configuration
IMAGE_SIZE = (64, 64)
SAMPLE_SIZE = 1000

# Paths
dataset_dir = Path("data/raw/Forest Segmented/Forest Segmented")
images_dir = dataset_dir / "images"
masks_dir = dataset_dir / "masks"
meta_file = dataset_dir / "meta_data.csv"
model_path = Path("data/models/deforestation_rf_model.joblib")

def extract_features(image):
    """Extract color histogram features"""
    # Calculate histograms for R, G, B channels
    hist_r = cv2.calcHist([image], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([image], [2], None, [32], [0, 256]).flatten()
    
    # Calculate mean and std dev
    mean, std = cv2.meanStdDev(image)
    
    # Concatenate all features
    features = np.concatenate([hist_r, hist_g, hist_b, mean.flatten(), std.flatten()])
    return features

print("\n🔄 Loading and processing images...")

meta_df = pd.read_csv(meta_file)
if len(meta_df) > SAMPLE_SIZE:
    meta_df = meta_df.sample(n=SAMPLE_SIZE, random_state=42)

X = []
y = []

start_time = time.time()
for idx, row in meta_df.iterrows():
    try:
        img_path = images_dir / row['image']
        mask_path = masks_dir / row['mask']
        
        if not img_path.exists(): continue
        
        # Read image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        
        # Extract features
        features = extract_features(img)
        X.append(features)
        
        # Read mask for label
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        forest_coverage = np.sum(mask > 0) / mask.size
        label = 0 if forest_coverage > 0.5 else 1  # 0=Forest, 1=Deforested
        y.append(label)
        
        if len(X) % 100 == 0:
            print(f"  Processed {len(X)} images...")
            
    except Exception:
        continue

X = np.array(X)
y = np.array(y)

print(f"✅ Processed {len(X)} images in {time.time()-start_time:.2f}s")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n🌲 Training Random Forest Model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n📊 Results:")
print(f"  - Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Deforested', 'Deforested']))

# Save
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(clf, model_path)
print(f"\n💾 Model saved to {model_path}")
