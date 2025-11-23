import numpy as np
from pathlib import Path
from typing import Union
import cv2
import joblib

class DeforestationPredictor:
    """Make predictions on satellite images using Random Forest model"""
    
    def __init__(self, model_path: str = None, image_size=(64, 64)):
        if model_path is None:
            # Default to RF model
            model_path = "data/models/deforestation_rf_model.joblib"
            
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.model = None
        self.class_names = ['Non-Deforested', 'Deforested']
        
        if self.model_path.exists():
            self.load_model()
        else:
            print(f"Warning: Model not found at {self.model_path}")
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = joblib.load(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features matching training (histograms + stats)"""
        # Calculate histograms for R, G, B channels
        hist_r = cv2.calcHist([image], [0], None, [32], [0, 256]).flatten()
        hist_g = cv2.calcHist([image], [1], None, [32], [0, 256]).flatten()
        hist_b = cv2.calcHist([image], [2], None, [32], [0, 256]).flatten()
        
        # Calculate mean and std dev
        mean, std = cv2.meanStdDev(image)
        
        # Concatenate all features
        features = np.concatenate([hist_r, hist_g, hist_b, mean.flatten(), std.flatten()])
        return features.reshape(1, -1)
    
    def preprocess_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """Preprocess image"""
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, self.image_size)
        return image
    
    def predict(self, image: Union[str, np.ndarray], return_probabilities=False):
        """Predict deforestation"""
        if self.model is None:
            return self._mock_predict(return_probabilities)
        
        # Preprocess
        processed_img = self.preprocess_image(image)
        features = self.extract_features(processed_img)
        
        # Predict
        if return_probabilities:
            probs = self.model.predict_proba(features)[0]
            predicted_class = np.argmax(probs)
            
            return {
                'probabilities': probs,
                'non_deforested_prob': float(probs[0]),
                'deforested_prob': float(probs[1]),
                'predicted_class': self.class_names[predicted_class],
                'confidence': float(np.max(probs))
            }
        else:
            prediction = self.model.predict(features)[0]
            return {
                'class': self.class_names[prediction],
                'class_id': int(prediction),
                'confidence': 1.0  # RF hard prediction
            }
            
    def _mock_predict(self, return_probabilities=False):
        """Fallback mock prediction"""
        print("Using mock prediction (Model not loaded)")
        return {
            'class': 'Non-Deforested',
            'class_id': 0,
            'confidence': 0.0,
            'deforested_prob': 0.0,
            'non_deforested_prob': 1.0
        }
    
    def get_risk_level(self, deforestation_probability: float) -> str:
        """Get risk level"""
        if deforestation_probability >= 0.8: return "Critical"
        elif deforestation_probability >= 0.6: return "High"
        elif deforestation_probability >= 0.4: return "Medium"
        elif deforestation_probability >= 0.2: return "Low"
        else: return "Minimal"
    
    def predict_with_risk(self, image: Union[str, np.ndarray]):
        """Predict with risk level"""
        result = self.predict(image, return_probabilities=True)
        result['risk_level'] = self.get_risk_level(result['deforested_prob'])
        return result
