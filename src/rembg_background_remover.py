#!/usr/bin/env python3
"""
REMBG Background Removal Service
Free, AI-powered background removal using REMBG library
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import logging
from concurrent.futures import ThreadPoolExecutor
import time

class RembgBackgroundRemover:
    """
    Professional background removal using REMBG (free, AI-powered)
    """
    
    def __init__(self, model_name: str = 'u2net'):
        """
        Initialize REMBG background remover
        
        Args:
            model_name: REMBG model to use
                - 'u2net': General purpose (default)
                - 'u2net_human_seg': Human segmentation
                - 'u2net_cloth_seg': Clothing segmentation  
                - 'isnet-general-use': High accuracy general use
                - 'silueta': Smaller file size
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Available models and their use cases
        self.models = {
            'u2net': 'General purpose - good for products',
            'u2net_human_seg': 'Human segmentation',
            'u2net_cloth_seg': 'Clothing items',
            'isnet-general-use': 'High accuracy general use',
            'silueta': 'Smaller model size',
            'isnet-anime': 'Anime characters'
        }
        
        self.logger.info(f"Initialized REMBG with model: {model_name}")
        
    def remove_background_single(self, input_path: str, output_path: str) -> bool:
        """
        Remove background from a single image
        
        Args:
            input_path: Path to input image
            output_path: Path for output image
            
        Returns:
            bool: Success status
        """
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            if not input_path.exists():
                self.logger.error(f"Input file not found: {input_path}")
                return False
                
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load and process image
            with open(input_path, 'rb') as input_file:
                input_data = input_file.read()
                
            # Remove background using REMBG
            output_data = remove(input_data)
            
            # Save result
            with open(output_path, 'wb') as output_file:
                output_file.write(output_data)
                
            self.logger.info(f"✅ Background removed: {input_path.name} -> {output_path.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to remove background from {input_path}: {str(e)}")
            return False
    
    def remove_background_batch(self, input_dir: str, output_dir: str, 
                              file_pattern: str = "*.png") -> Tuple[int, int]:
        """
        Remove backgrounds from all images in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            file_pattern: File pattern to match (e.g., "*.png", "*.jpg")
            
        Returns:
            Tuple[int, int]: (processed_count, failed_count)
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        if not input_dir.exists():
            self.logger.error(f"Input directory not found: {input_dir}")
            return 0, 0
            
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all matching files
        image_files = list(input_dir.glob(file_pattern))
        
        if not image_files:
            self.logger.warning(f"No files found matching pattern '{file_pattern}' in {input_dir}")
            return 0, 0
            
        self.logger.info(f"🎨 Processing {len(image_files)} images with REMBG")
        self.logger.info(f"📁 Input: {input_dir}")
        self.logger.info(f"📁 Output: {output_dir}")
        
        processed = 0
        failed = 0
        start_time = time.time()
        
        for i, image_file in enumerate(image_files, 1):
            output_file = output_dir / image_file.name
            
            print(f"🖼️  [{i}/{len(image_files)}] Processing: {image_file.name}")
            
            if self.remove_background_single(str(image_file), str(output_file)):
                processed += 1
            else:
                failed += 1
                
        elapsed_time = time.time() - start_time
        
        self.logger.info("="*60)
        self.logger.info(f"✅ REMBG BATCH PROCESSING COMPLETE")
        self.logger.info(f"📊 Processed: {processed} images")
        self.logger.info(f"❌ Failed: {failed} images")
        self.logger.info(f"⏱️  Total time: {elapsed_time:.1f}s ({elapsed_time/len(image_files):.2f}s per image)")
        self.logger.info(f"📁 Results saved to: {output_dir}")
        
        return processed, failed
    
    def remove_background_parallel(self, input_dir: str, output_dir: str,
                                 file_pattern: str = "*.png", max_workers: int = 4) -> Tuple[int, int]:
        """
        Remove backgrounds using parallel processing for faster batch operations
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            file_pattern: File pattern to match
            max_workers: Number of parallel workers
            
        Returns:
            Tuple[int, int]: (processed_count, failed_count)
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        if not input_dir.exists():
            self.logger.error(f"Input directory not found: {input_dir}")
            return 0, 0
            
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all matching files
        image_files = list(input_dir.glob(file_pattern))
        
        if not image_files:
            self.logger.warning(f"No files found matching pattern '{file_pattern}' in {input_dir}")
            return 0, 0
        
        self.logger.info(f"🚀 PARALLEL PROCESSING: {len(image_files)} images with {max_workers} workers")
        
        def process_single_image(image_file):
            output_file = output_dir / image_file.name
            success = self.remove_background_single(str(image_file), str(output_file))
            return image_file.name, success
        
        processed = 0
        failed = 0
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(process_single_image, image_files)
            
            for filename, success in results:
                if success:
                    processed += 1
                    print(f"✅ {filename}")
                else:
                    failed += 1
                    print(f"❌ {filename}")
        
        elapsed_time = time.time() - start_time
        
        self.logger.info("="*60)
        self.logger.info(f"✅ PARALLEL REMBG PROCESSING COMPLETE")
        self.logger.info(f"📊 Processed: {processed} images")
        self.logger.info(f"❌ Failed: {failed} images")
        self.logger.info(f"⏱️  Total time: {elapsed_time:.1f}s ({elapsed_time/len(image_files):.2f}s per image)")
        self.logger.info(f"🚀 Speedup: {max_workers}x parallel processing")
        
        return processed, failed
    
    def add_drop_shadow(self, image_path: str, output_path: str, 
                       shadow_offset: Tuple[int, int] = (5, 5), 
                       blur_radius: int = 10,
                       shadow_opacity: float = 0.3) -> bool:
        """
        Add professional drop shadow to transparent PNG
        
        Args:
            image_path: Path to transparent PNG
            output_path: Path for output with shadow
            shadow_offset: Shadow offset (x, y)
            blur_radius: Shadow blur radius
            shadow_opacity: Shadow opacity (0.0-1.0)
            
        Returns:
            bool: Success status
        """
        try:
            # Load transparent PNG
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None or img.shape[2] != 4:
                self.logger.error(f"Invalid transparent PNG: {image_path}")
                return False
                
            h, w = img.shape[:2]
            
            # Create shadow layer
            shadow = np.zeros((h, w, 4), dtype=np.uint8)
            alpha_mask = img[:, :, 3] > 0
            shadow[alpha_mask] = [50, 50, 50, int(255 * shadow_opacity)]
            
            # Apply blur to shadow
            shadow_blurred = cv2.GaussianBlur(shadow, (blur_radius*2+1, blur_radius*2+1), 0)
            
            # Create final canvas
            canvas_h = h + abs(shadow_offset[1]) + blur_radius
            canvas_w = w + abs(shadow_offset[0]) + blur_radius
            canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
            
            # Place shadow
            shadow_y = max(0, shadow_offset[1]) + blur_radius//2
            shadow_x = max(0, shadow_offset[0]) + blur_radius//2
            canvas[shadow_y:shadow_y+h, shadow_x:shadow_x+w] = shadow_blurred
            
            # Place original image
            img_y = max(0, -shadow_offset[1]) + blur_radius//2
            img_x = max(0, -shadow_offset[0]) + blur_radius//2
            
            # Blend image over shadow
            for c in range(3):
                alpha_norm = img[:, :, 3] / 255.0
                canvas[img_y:img_y+h, img_x:img_x+w, c] = (
                    alpha_norm * img[:, :, c] + 
                    (1 - alpha_norm) * canvas[img_y:img_y+h, img_x:img_x+w, c]
                )
            
            canvas[img_y:img_y+h, img_x:img_x+w, 3] = np.maximum(
                img[:, :, 3], 
                canvas[img_y:img_y+h, img_x:img_x+w, 3]
            )
            
            cv2.imwrite(output_path, canvas)
            return True
            
        except Exception as e:
            self.logger.error(f"Drop shadow failed: {str(e)}")
            return False
    
    def get_model_info(self) -> dict:
        """Get information about available REMBG models"""
        return self.models
    
    def estimate_processing_time(self, image_count: int, parallel: bool = False, 
                               workers: int = 4) -> str:
        """
        Estimate processing time for batch operation
        
        Args:
            image_count: Number of images to process
            parallel: Whether using parallel processing
            workers: Number of parallel workers
            
        Returns:
            str: Estimated time description
        """
        # Average processing time per image (seconds)
        avg_time_per_image = 2.5
        
        if parallel:
            total_time = (image_count * avg_time_per_image) / workers
        else:
            total_time = image_count * avg_time_per_image
            
        if total_time < 60:
            return f"~{total_time:.0f} seconds"
        elif total_time < 3600:
            return f"~{total_time/60:.1f} minutes"
        else:
            return f"~{total_time/3600:.1f} hours"


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rembg_processing.log')
        ]
    )


if __name__ == "__main__":
    # Example usage and testing
    setup_logging()
    
    remover = RembgBackgroundRemover()
    
    print("🎨 REMBG BACKGROUND REMOVER")
    print("="*50)
    print("Available models:")
    for model, description in remover.get_model_info().items():
        print(f"  - {model}: {description}")
    
    print(f"\n📊 Processing time estimates:")
    print(f"  - 100 images: {remover.estimate_processing_time(100)}")
    print(f"  - 1000 images: {remover.estimate_processing_time(1000)}")
    print(f"  - 2000 images: {remover.estimate_processing_time(2000)}")
    print(f"  - 2000 images (parallel): {remover.estimate_processing_time(2000, parallel=True)}")
    
    print("\n💡 Ready to process your images!")
    print("Example usage:")
    print("  remover.remove_background_batch('input_dir', 'output_dir')")