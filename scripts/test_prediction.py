import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.predictor import DeforestationPredictor
import cv2
import numpy as np

# Create a dummy image
dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
cv2.imwrite("test_image.jpg", dummy_img)

try:
    print("Initializing predictor...")
    predictor = DeforestationPredictor()
    
    print("Making prediction...")
    result = predictor.predict_with_risk("test_image.jpg")
    
    print("\n✅ Prediction Successful!")
    print(f"Result: {result}")
except Exception as e:
    print(f"\n❌ Prediction Failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    import os
    if os.path.exists("test_image.jpg"):
        os.remove("test_image.jpg")
