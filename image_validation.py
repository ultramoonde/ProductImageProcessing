#!/usr/bin/env python3
"""
Image Validation Module
Prevents corrupted/garbage image output in the pipeline
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import hashlib


class ImageValidator:
    """Comprehensive image validation to prevent corrupted outputs"""

    def __init__(self):
        self.min_file_size = 1024  # 1KB minimum
        self.max_file_size = 50 * 1024 * 1024  # 50MB maximum
        self.min_dimensions = (50, 50)  # 50x50 minimum
        self.max_dimensions = (8192, 8192)  # 8K maximum

    def validate_image_data(self, image: np.ndarray, description: str = "image") -> bool:
        """Validate numpy array image data before processing"""
        try:
            if image is None:
                print(f"❌ {description}: Image is None")
                return False

            if not isinstance(image, np.ndarray):
                print(f"❌ {description}: Not a numpy array")
                return False

            if image.size == 0:
                print(f"❌ {description}: Empty image array")
                return False

            if len(image.shape) not in [2, 3]:
                print(f"❌ {description}: Invalid dimensions {image.shape}")
                return False

            height, width = image.shape[:2]
            if height < self.min_dimensions[0] or width < self.min_dimensions[1]:
                print(f"❌ {description}: Too small {width}x{height}")
                return False

            if height > self.max_dimensions[0] or width > self.max_dimensions[1]:
                print(f"❌ {description}: Too large {width}x{height}")
                return False

            # Check for all-zero or all-same pixel corruption
            if np.all(image == 0):
                print(f"❌ {description}: All black pixels (corrupted)")
                return False

            if len(image.shape) == 3 and image.shape[2] >= 3:
                # Check for uniform color corruption
                flat = image.reshape(-1, image.shape[2])
                if len(np.unique(flat, axis=0)) < 5:
                    print(f"❌ {description}: Too few unique colors (corrupted)")
                    return False

            # Check data type validity
            if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
                print(f"❌ {description}: Invalid data type {image.dtype}")
                return False

            print(f"✅ {description}: Valid image data {width}x{height}")
            return True

        except Exception as e:
            print(f"❌ {description}: Validation error: {e}")
            return False

    def validate_file_before_save(self, file_path: str, image: np.ndarray) -> bool:
        """Validate image before saving to file"""
        if not self.validate_image_data(image, f"Pre-save {Path(file_path).name}"):
            return False

        # Check file path validity
        try:
            path = Path(file_path)
            if not path.parent.exists():
                print(f"❌ Directory doesn't exist: {path.parent}")
                return False

            # Check for suspicious file extensions
            valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
            if path.suffix.lower() not in valid_extensions:
                print(f"❌ Invalid file extension: {path.suffix}")
                return False

            return True

        except Exception as e:
            print(f"❌ File path validation error: {e}")
            return False

    def validate_saved_file(self, file_path: str) -> Dict[str, Any]:
        """Validate file after saving"""
        result = {
            "valid": False,
            "file_size": 0,
            "dimensions": None,
            "channels": 0,
            "hash": None,
            "errors": []
        }

        try:
            if not os.path.exists(file_path):
                result["errors"].append("File doesn't exist")
                return result

            # Check file size
            file_size = os.path.getsize(file_path)
            result["file_size"] = file_size

            if file_size < self.min_file_size:
                result["errors"].append(f"File too small: {file_size} bytes")
                return result

            if file_size > self.max_file_size:
                result["errors"].append(f"File too large: {file_size} bytes")
                return result

            # Try to load the image
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                result["errors"].append("Cannot load image with OpenCV")
                return result

            # Validate loaded image
            if not self.validate_image_data(image, f"Saved {Path(file_path).name}"):
                result["errors"].append("Image data validation failed")
                return result

            result["dimensions"] = (image.shape[1], image.shape[0])  # width, height
            result["channels"] = image.shape[2] if len(image.shape) == 3 else 1

            # Generate file hash for integrity checking
            with open(file_path, 'rb') as f:
                result["hash"] = hashlib.md5(f.read()).hexdigest()

            result["valid"] = True
            print(f"✅ Saved file validation passed: {Path(file_path).name}")

        except Exception as e:
            result["errors"].append(f"Validation exception: {e}")
            print(f"❌ Saved file validation failed: {e}")

        return result

    def safe_image_save(self, image: np.ndarray, file_path: str, quality: int = 95) -> bool:
        """Safely save image with comprehensive validation"""
        try:
            # Pre-save validation
            if not self.validate_file_before_save(file_path, image):
                print(f"❌ Pre-save validation failed for {file_path}")
                return False

            # Save the image
            path = Path(file_path)
            if path.suffix.lower() == '.png':
                success = cv2.imwrite(str(file_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            elif path.suffix.lower() in ['.jpg', '.jpeg']:
                success = cv2.imwrite(str(file_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                success = cv2.imwrite(str(file_path), image)

            if not success:
                print(f"❌ OpenCV write failed for {file_path}")
                return False

            # Post-save validation
            validation_result = self.validate_saved_file(file_path)
            if not validation_result["valid"]:
                print(f"❌ Post-save validation failed: {validation_result['errors']}")
                # Remove corrupted file
                try:
                    os.remove(file_path)
                    print(f"🗑️ Removed corrupted file: {file_path}")
                except:
                    pass
                return False

            print(f"✅ Successfully saved and validated: {Path(file_path).name}")
            print(f"   📏 Size: {validation_result['file_size']} bytes")
            print(f"   📐 Dimensions: {validation_result['dimensions']}")
            print(f"   🎨 Channels: {validation_result['channels']}")

            return True

        except Exception as e:
            print(f"❌ Safe save failed for {file_path}: {e}")
            return False

    def cleanup_test_artifacts(self, directory: str, patterns: list = None) -> int:
        """Clean up test files and artifacts that may contain corrupted data"""
        if patterns is None:
            patterns = ['test_*.png', 'debug_*.png', 'temp_*.png', '*_test.png']

        cleaned = 0
        try:
            for pattern in patterns:
                for file_path in Path(directory).glob(pattern):
                    try:
                        # Validate if it's actually corrupted
                        validation = self.validate_saved_file(str(file_path))
                        if not validation["valid"]:
                            file_path.unlink()
                            print(f"🗑️ Removed corrupted test file: {file_path.name}")
                            cleaned += 1
                        else:
                            print(f"✅ Test file is valid, keeping: {file_path.name}")
                    except Exception as e:
                        print(f"⚠️ Error checking {file_path}: {e}")

        except Exception as e:
            print(f"❌ Cleanup error: {e}")

        return cleaned


def create_validation_wrapper():
    """Create a validation decorator for image processing functions"""
    validator = ImageValidator()

    def validate_image_processing(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            # If result contains image data, validate it
            if isinstance(result, dict) and 'image' in result:
                if not validator.validate_image_data(result['image'], f"{func.__name__} output"):
                    result['success'] = False
                    result['error'] = "Output validation failed"
                    result['image'] = None

            return result
        return wrapper
    return validate_image_processing


# Global validator instance
validator = ImageValidator()

if __name__ == "__main__":
    # Test the validator
    print("🧪 Testing Image Validator")

    # Test with a real image if available
    test_dir = "/Users/davemooney/_dev/pngImageExtraction/latest/step_by_step_flat"
    if os.path.exists(test_dir):
        cleaned = validator.cleanup_test_artifacts(test_dir)
        print(f"🗑️ Cleaned up {cleaned} corrupted test files")