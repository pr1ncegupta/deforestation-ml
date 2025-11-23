import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

class ModelTrainer:
    """Train and evaluate deforestation detection model"""
    
    def __init__(self, model, model_save_path):
        self.model = model
        self.model_save_path = Path(model_save_path)
        self.history = None
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, verbose=1):
        """
        Train the model
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        # Create callbacks
        callbacks = self._create_callbacks()
        
        # Train model
        print(f"Starting training for {epochs} epochs...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("\nTraining completed!")
        self._save_training_history()
        
        return self.history
    
    def _create_callbacks(self):
        """Create training callbacks"""
        callbacks = [
            # Save best model
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_save_path),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir=str(self.model_save_path.parent / 'logs'),
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def evaluate(self, X_test, y_test, batch_size=32):
        """
        Evaluate model on test set
        
        Args:
            X_test: Test images
            y_test: Test labels
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\nEvaluating model on test set...")
        
        results = self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
        
        metrics = {
            'test_loss': results[0],
            'test_accuracy': results[1],
            'test_precision': results[2] if len(results) > 2 else None,
            'test_recall': results[3] if len(results) > 3 else None
        }
        
        # Calculate F1 score if precision and recall are available
        if metrics['test_precision'] and metrics['test_recall']:
            metrics['test_f1'] = 2 * (metrics['test_precision'] * metrics['test_recall']) / \
                                (metrics['test_precision'] + metrics['test_recall'])
        
        print("\n" + "="*50)
        print("TEST SET RESULTS")
        print("="*50)
        for metric, value in metrics.items():
            if value is not None:
                print(f"{metric}: {value:.4f}")
        print("="*50)
        
        # Save evaluation results
        self._save_evaluation_results(metrics)
        
        return metrics
    
    def predict(self, X, batch_size=32):
        """
        Make predictions
        
        Args:
            X: Input images
            batch_size: Batch size
            
        Returns:
            Predictions
        """
        return self.model.predict(X, batch_size=batch_size)
    
    def _save_training_history(self):
        """Save training history to JSON"""
        if self.history is None:
            return
        
        history_path = self.model_save_path.parent / 'training_history.json'
        
        # Convert history to serializable format
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"Training history saved to {history_path}")
    
    def _save_evaluation_results(self, metrics):
        """Save evaluation results to JSON"""
        results_path = self.model_save_path.parent / 'evaluation_results.json'
        
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Evaluation results saved to {results_path}")
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot precision if available
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Train')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot recall if available
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Train')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        else:
            plt.savefig(self.model_save_path.parent / 'training_history.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def load_model(self, model_path=None):
        """
        Load a saved model
        
        Args:
            model_path: Path to model file (uses default if None)
        """
        if model_path is None:
            model_path = self.model_save_path
        
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        
        return self.model
