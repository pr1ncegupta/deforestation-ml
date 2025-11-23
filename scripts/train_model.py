#!/usr/bin/env python3
"""
Train deforestation detection model
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import SatelliteDataLoader
from src.data.preprocessor import ImagePreprocessor
from src.models.cnn_model import DeforestationCNN
from src.models.trainer import ModelTrainer
import config

def main():
    print("="*60)
    print("DEFORESTATION DETECTION MODEL TRAINING")
    print("="*60)
    
    # Configuration
    print("\n📋 Configuration:")
    print(f"  - Image size: {config.IMAGE_SIZE}")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Epochs: {config.EPOCHS}")
    print(f"  - Learning rate: {config.LEARNING_RATE}")
    print(f"  - Model name: {config.MODEL_NAME}")
    
    # Initialize components
    print("\n🔧 Initializing components...")
    data_loader = SatelliteDataLoader(config.RAW_DATA_DIR)
    preprocessor = ImagePreprocessor(target_size=config.IMAGE_SIZE)
    
    # Load data
    print("\n📂 Loading dataset...")
    try:
        images, labels = data_loader.load_dataset()
        print(f"  ✅ Loaded {len(images)} images")
        print(f"  - Class distribution: {np.bincount(labels)}")
    except Exception as e:
        print(f"  ❌ Error loading dataset: {e}")
        print("\n💡 Note: Make sure you have downloaded the dataset first!")
        print("  Run: python scripts/download_data.py")
        return
    
    # Preprocess data
    print("\n🔄 Preprocessing dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_dataset(
        images, labels,
        test_size=0.2,
        val_size=0.1
    )
    
    # Optional: Create augmented dataset
    print("\n🎨 Creating augmented dataset...")
    X_train_aug, y_train_aug = preprocessor.create_augmented_dataset(
        X_train, y_train,
        augmentation_factor=2
    )
    
    # Build model
    print("\n🏗️ Building model...")
    cnn = DeforestationCNN(
        input_shape=(*config.IMAGE_SIZE, 3),
        num_classes=config.NUM_CLASSES
    )
    
    # Choose architecture
    architecture = 'standard'  # Options: 'standard', 'deep', 'resnet'
    cnn.build_model(architecture=architecture)
    cnn.compile_model(learning_rate=config.LEARNING_RATE)
    
    print(f"  ✅ Model built with '{architecture}' architecture")
    print("\n📊 Model Summary:")
    cnn.summary()
    
    # Initialize trainer
    model_path = config.MODELS_DIR / f"{config.MODEL_NAME}.h5"
    trainer = ModelTrainer(
        model=cnn.get_model(),
        model_save_path=str(model_path)
    )
    
    # Train model
    print("\n🚀 Starting training...")
    print("="*60)
    
    history = trainer.train(
        X_train_aug, y_train_aug,
        X_val, y_val,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    metrics = trainer.evaluate(X_test, y_test)
    
    # Plot training history
    print("\n📈 Generating training plots...")
    trainer.plot_training_history()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📁 Model saved to: {model_path}")
    print(f"📁 Training history: {model_path.parent / 'training_history.json'}")
    print(f"📁 Evaluation results: {model_path.parent / 'evaluation_results.json'}")
    print(f"📁 Training plot: {model_path.parent / 'training_history.png'}")
    print("\n🎉 You can now use the model for predictions!")

if __name__ == "__main__":
    main()
