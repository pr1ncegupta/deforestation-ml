import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from typing import Tuple
import albumentations as A

class ImagePreprocessor:
    """Preprocess satellite images for model training"""
    
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
        self.augmentation_pipeline = self._create_augmentation_pipeline()
    
    def _create_augmentation_pipeline(self):
        """Create augmentation pipeline using albumentations"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
        ])
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image array
            
        Returns:
            Resized image
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [0, 1]
        
        Args:
            image: Input image array
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to image
        
        Args:
            image: Input image array
            
        Returns:
            Augmented image
        """
        # Ensure image is uint8 for albumentations
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        augmented = self.augmentation_pipeline(image=image)
        return augmented['image']
    
    def augment_image_simple(self, image: np.ndarray) -> np.ndarray:
        """
        Apply simple augmentation without albumentations (fallback)
        
        Args:
            image: Input image array
            
        Returns:
            Augmented image
        """
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
        
        # Random vertical flip
        if np.random.rand() > 0.5:
            image = np.flipud(image)
        
        # Random rotation (90, 180, 270 degrees)
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k)
        
        # Random brightness adjustment
        if np.random.rand() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 255).astype(image.dtype)
        
        return image
    
    def preprocess_single_image(self, image: np.ndarray, augment: bool = False) -> np.ndarray:
        """
        Preprocess a single image
        
        Args:
            image: Input image array
            augment: Whether to apply augmentation
            
        Returns:
            Preprocessed image
        """
        # Resize
        image = self.resize_image(image)
        
        # Augment if requested
        if augment:
            try:
                image = self.augment_image(image)
            except:
                # Fallback to simple augmentation
                image = self.augment_image_simple(image)
        
        # Normalize
        image = self.normalize_image(image)
        
        return image
    
    def preprocess_dataset(self, images: np.ndarray, labels: np.ndarray, 
                          test_size: float = 0.2, val_size: float = 0.1,
                          augment_train: bool = True) -> Tuple:
        """
        Preprocess entire dataset and split into train/val/test
        
        Args:
            images: Array of images
            labels: Array of labels
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            augment_train: Whether to augment training data
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        processed_images = []
        
        print(f"Preprocessing {len(images)} images...")
        
        for i, img in enumerate(images):
            if i % 100 == 0:
                print(f"Processed {i}/{len(images)} images")
            
            # Preprocess without augmentation initially
            img = self.preprocess_single_image(img, augment=False)
            processed_images.append(img)
        
        processed_images = np.array(processed_images)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            processed_images, labels, 
            test_size=test_size, 
            random_state=42,
            stratify=labels
        )
        
        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=y_temp
        )
        
        print(f"\nDataset split:")
        print(f"Training: {len(X_train)} images")
        print(f"Validation: {len(X_val)} images")
        print(f"Testing: {len(X_test)} images")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_augmented_dataset(self, X_train: np.ndarray, y_train: np.ndarray, 
                                augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create augmented training dataset
        
        Args:
            X_train: Training images
            y_train: Training labels
            augmentation_factor: How many augmented versions to create per image
            
        Returns:
            Augmented (X_train, y_train)
        """
        augmented_images = list(X_train)
        augmented_labels = list(y_train)
        
        print(f"Creating {augmentation_factor}x augmented dataset...")
        
        for i in range(augmentation_factor - 1):
            for img, label in zip(X_train, y_train):
                # Denormalize for augmentation
                img_uint8 = (img * 255).astype(np.uint8)
                
                # Augment
                try:
                    aug_img = self.augment_image(img_uint8)
                except:
                    aug_img = self.augment_image_simple(img_uint8)
                
                # Normalize again
                aug_img = self.normalize_image(aug_img)
                
                augmented_images.append(aug_img)
                augmented_labels.append(label)
        
        print(f"Augmented dataset size: {len(augmented_images)} images")
        
        return np.array(augmented_images), np.array(augmented_labels)
