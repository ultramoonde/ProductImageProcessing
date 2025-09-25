#!/usr/bin/env python3
"""
Professional Background Removal Service
Supports multiple AI-powered background removal APIs
"""

import requests
import os
import base64
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
import numpy as np


class BackgroundRemovalService:
    """
    Professional background removal using AI APIs
    """
    
    def __init__(self):
        self.apis = {
            'removal_ai': {
                'url': 'https://api.removal.ai/4.0/remove',
                'api_key_env': 'REMOVAL_AI_API_KEY',
                'cost_per_image': 0.15
            },
            'remove_bg': {
                'url': 'https://api.remove.bg/v1.0/removebg',
                'api_key_env': 'REMOVE_BG_API_KEY', 
                'cost_per_image': 0.20
            },
            'photoroom': {
                'url': 'https://image-api.photoroom.com/v1/segment',
                'api_key_env': 'PHOTOROOM_API_KEY',
                'cost_per_image': 0.10
            }
        }
        
    def remove_background_removal_ai(self, image_path: str, output_path: str) -> bool:
        """Remove background using Removal.AI API"""
        api_key = os.getenv('REMOVAL_AI_API_KEY')
        if not api_key:
            print("❌ REMOVAL_AI_API_KEY not set")
            return False
            
        try:
            with open(image_path, 'rb') as image_file:
                files = {'image_file': image_file}
                headers = {'Api-Key': api_key}
                
                response = requests.post(
                    'https://api.removal.ai/4.0/remove',
                    files=files,
                    headers=headers,
                    timeout=30
                )
                
            if response.status_code == 200:
                with open(output_path, 'wb') as output_file:
                    output_file.write(response.content)
                print(f"✅ Removal.AI: Background removed successfully")
                return True
            else:
                print(f"❌ Removal.AI API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Removal.AI error: {str(e)}")
            return False
    
    def remove_background_remove_bg(self, image_path: str, output_path: str) -> bool:
        """Remove background using Remove.bg API"""
        api_key = os.getenv('REMOVE_BG_API_KEY')
        if not api_key:
            print("❌ REMOVE_BG_API_KEY not set")
            return False
            
        try:
            with open(image_path, 'rb') as image_file:
                response = requests.post(
                    'https://api.remove.bg/v1.0/removebg',
                    files={'image_file': image_file},
                    data={'size': 'auto'},
                    headers={'X-Api-Key': api_key},
                    timeout=30
                )
                
            if response.status_code == 200:
                with open(output_path, 'wb') as output_file:
                    output_file.write(response.content)
                print(f"✅ Remove.bg: Background removed successfully")
                return True
            else:
                print(f"❌ Remove.bg API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Remove.bg error: {str(e)}")
            return False
    
    def remove_background_photoroom(self, image_path: str, output_path: str) -> bool:
        """Remove background using Photoroom API"""
        api_key = os.getenv('PHOTOROOM_API_KEY')
        if not api_key:
            print("❌ PHOTOROOM_API_KEY not set")
            return False
            
        try:
            with open(image_path, 'rb') as image_file:
                files = {'image_file': image_file}
                headers = {'x-api-key': api_key}
                
                response = requests.post(
                    'https://image-api.photoroom.com/v1/segment',
                    files=files,
                    headers=headers,
                    timeout=30
                )
                
            if response.status_code == 200:
                with open(output_path, 'wb') as output_file:
                    output_file.write(response.content)
                print(f"✅ Photoroom: Background removed successfully")
                return True
            else:
                print(f"❌ Photoroom API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Photoroom error: {str(e)}")
            return False
    
    def remove_background(self, image_path: str, output_path: str, 
                         preferred_service: str = 'removal_ai') -> bool:
        """
        Remove background using preferred service with fallbacks
        
        Args:
            image_path: Path to input image
            output_path: Path for output image  
            preferred_service: 'removal_ai', 'remove_bg', or 'photoroom'
            
        Returns:
            bool: Success status
        """
        print(f"🎨 Removing background from {Path(image_path).name}")
        
        # Try preferred service first
        if preferred_service == 'removal_ai':
            if self.remove_background_removal_ai(image_path, output_path):
                return True
        elif preferred_service == 'remove_bg':
            if self.remove_background_remove_bg(image_path, output_path):
                return True
        elif preferred_service == 'photoroom':
            if self.remove_background_photoroom(image_path, output_path):
                return True
        
        # Try fallback services
        print("⚠️  Trying fallback services...")
        
        services = ['removal_ai', 'remove_bg', 'photoroom']
        if preferred_service in services:
            services.remove(preferred_service)
            
        for service in services:
            if service == 'removal_ai':
                if self.remove_background_removal_ai(image_path, output_path):
                    return True
            elif service == 'remove_bg':
                if self.remove_background_remove_bg(image_path, output_path):
                    return True
            elif service == 'photoroom':
                if self.remove_background_photoroom(image_path, output_path):
                    return True
                    
        print("❌ All background removal services failed")
        return False
    
    def add_drop_shadow(self, image_path: str, output_path: str, 
                       shadow_offset: tuple = (5, 5), blur_radius: int = 10,
                       shadow_opacity: float = 0.3) -> bool:
        """Add professional drop shadow to transparent PNG"""
        try:
            # Load image with alpha channel
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return False
                
            if img.shape[2] != 4:
                print("⚠️  Image doesn't have alpha channel")
                return False
                
            h, w = img.shape[:2]
            
            # Create shadow layer
            shadow = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Use alpha channel as shadow base
            alpha = img[:, :, 3]
            shadow_mask = alpha > 0
            
            # Create shadow
            shadow[shadow_mask] = [50, 50, 50, int(255 * shadow_opacity)]  # Dark gray shadow
            
            # Apply Gaussian blur to shadow
            shadow_blurred = cv2.GaussianBlur(shadow, (blur_radius*2+1, blur_radius*2+1), 0)
            
            # Create final canvas with shadow offset
            canvas_h = h + abs(shadow_offset[1]) + blur_radius
            canvas_w = w + abs(shadow_offset[0]) + blur_radius
            canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
            
            # Place shadow
            shadow_y = max(0, shadow_offset[1]) + blur_radius//2
            shadow_x = max(0, shadow_offset[0]) + blur_radius//2
            canvas[shadow_y:shadow_y+h, shadow_x:shadow_x+w] = shadow_blurred
            
            # Place original image on top
            img_y = max(0, -shadow_offset[1]) + blur_radius//2
            img_x = max(0, -shadow_offset[0]) + blur_radius//2
            
            # Blend original image over shadow
            for c in range(3):
                alpha_norm = img[:, :, 3] / 255.0
                canvas[img_y:img_y+h, img_x:img_x+w, c] = (
                    alpha_norm * img[:, :, c] + 
                    (1 - alpha_norm) * canvas[img_y:img_y+h, img_x:img_x+w, c]
                )
            
            # Set alpha for original image area
            canvas[img_y:img_y+h, img_x:img_x+w, 3] = np.maximum(
                img[:, :, 3], 
                canvas[img_y:img_y+h, img_x:img_x+w, 3]
            )
            
            cv2.imwrite(output_path, canvas)
            return True
            
        except Exception as e:
            print(f"❌ Drop shadow error: {str(e)}")
            return False
    
    def process_product_image(self, image_path: str, output_dir: str = None,
                            add_shadow: bool = True, 
                            preferred_service: str = 'removal_ai') -> Optional[str]:
        """
        Complete product image processing pipeline
        
        Args:
            image_path: Input image path
            output_dir: Output directory (defaults to same as input)
            add_shadow: Whether to add drop shadow
            preferred_service: Background removal service to use
            
        Returns:
            str: Path to processed image or None if failed
        """
        input_path = Path(image_path)
        if not input_path.exists():
            print(f"❌ Input image not found: {image_path}")
            return None
            
        # Set output directory
        if output_dir is None:
            output_dir = input_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
        # Process image
        base_name = input_path.stem
        bg_removed_path = output_dir / f"{base_name}_no_bg.png"
        final_path = output_dir / f"{base_name}_final.png"
        
        # Remove background
        if not self.remove_background(str(input_path), str(bg_removed_path), preferred_service):
            return None
            
        # Add drop shadow if requested
        if add_shadow:
            if self.add_drop_shadow(str(bg_removed_path), str(final_path)):
                return str(final_path)
            else:
                print("⚠️  Shadow addition failed, returning background-removed image")
                return str(bg_removed_path)
        else:
            return str(bg_removed_path)


def setup_api_keys_instructions():
    """Print instructions for setting up API keys"""
    print("""
🔑 BACKGROUND REMOVAL API SETUP INSTRUCTIONS

To use professional background removal services, set these environment variables:

1. REMOVAL.AI (Recommended - $0.15/image):
   export REMOVAL_AI_API_KEY="your_api_key_here"
   Sign up at: https://removal.ai/
   
2. REMOVE.BG (Industry Standard - $0.20/image):
   export REMOVE_BG_API_KEY="your_api_key_here"
   Sign up at: https://www.remove.bg/api
   
3. PHOTOROOM (Product Focused - $0.10/image):
   export PHOTOROOM_API_KEY="your_api_key_here"  
   Sign up at: https://www.photoroom.com/api/

Example usage:
    service = BackgroundRemovalService()
    result = service.process_product_image('product.jpg', add_shadow=True)
    
Cost comparison per 1000 images:
- Photoroom: $100
- Removal.AI: $150  
- Remove.bg: $200
""")


if __name__ == "__main__":
    setup_api_keys_instructions()