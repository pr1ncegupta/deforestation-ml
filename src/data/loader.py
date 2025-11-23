import os
import numpy as np
from PIL import Image
from pathlib import Path
import rasterio
from typing import Tuple, List
import cv2

class SatelliteDataLoader:
    """Load and manage satellite imagery data"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load satellite image from file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy array of the image
        """
        try:
            # Try loading as geospatial raster
            with rasterio.open(image_path) as src:
                image = src.read()
                # Transpose to (H, W, C)
                image = np.transpose(image, (1, 2, 0))
        except:
            # Fall back to regular image loading
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        return image
    
    def load_images_from_directory(self, directory: str, extensions: List[str] = None) -> List[np.ndarray]:
        """
        Load all images from a directory
        
        Args:
            directory: Path to directory containing images
            extensions: List of file extensions to load (default: ['.jpg', '.png', '.tif'])
            
        Returns:
            List of image arrays
        """
        if extensions is None:
            extensions = ['.jpg', '.png', '.tif', '.tiff']
            
        directory = Path(directory)
        images = []
        
        for ext in extensions:
            for img_path in directory.glob(f'*{ext}'):
                try:
                    img = self.load_image(str(img_path))
                    images.append(img)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    
        return images
    
    def load_dataset(self, labels_file: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load entire dataset with labels
        
        Args:
            labels_file: Path to CSV file containing labels
            
        Returns:
            Tuple of (images, labels)
        """
        images = []
        labels = []
        
        # Load images from subdirectories (deforested/non-deforested)
        deforested_dir = self.data_dir / "deforested"
        non_deforested_dir = self.data_dir / "non_deforested"
        
        if deforested_dir.exists():
            deforested_images = self.load_images_from_directory(deforested_dir)
            images.extend(deforested_images)
            labels.extend([1] * len(deforested_images))
            
        if non_deforested_dir.exists():
            non_deforested_images = self.load_images_from_directory(non_deforested_dir)
            images.extend(non_deforested_images)
            labels.extend([0] * len(non_deforested_images))
        
        return np.array(images), np.array(labels)
    
    def get_image_info(self, image_path: str) -> dict:
        """
        Get metadata about an image
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with image metadata
        """
        try:
            with rasterio.open(image_path) as src:
                return {
                    'width': src.width,
                    'height': src.height,
                    'bands': src.count,
                    'dtype': src.dtypes[0],
                    'crs': src.crs,
                    'bounds': src.bounds
                }
        except:
            img = Image.open(image_path)
            return {
                'width': img.width,
                'height': img.height,
                'mode': img.mode,
                'format': img.format
            }
