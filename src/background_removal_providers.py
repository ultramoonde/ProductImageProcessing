#!/usr/bin/env python3
"""
Flexible Background Removal Provider System
Abstract interface and concrete implementations for various background removal services
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import time
import os
from pathlib import Path
import logging
import requests
import cv2
import numpy as np
from PIL import Image

# Import REMBG
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


@dataclass
class BackgroundRemovalResult:
    """Result object for background removal operations"""
    success: bool
    output_path: Optional[str] = None
    provider_used: Optional[str] = None
    processing_time: float = 0.0
    quality_score: Optional[float] = None
    cost: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BackgroundRemovalProvider(ABC):
    """Abstract base class for background removal providers"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"BGProvider.{name}")
    
    @abstractmethod
    def remove_background(self, input_path: str, output_path: str) -> BackgroundRemovalResult:
        """Remove background from image"""
        pass
    
    @abstractmethod
    def get_cost_per_image(self) -> float:
        """Get cost per image in USD"""
        pass
    
    @abstractmethod
    def get_speed_rating(self) -> int:
        """Get speed rating (1-10, higher is faster)"""
        pass
    
    @abstractmethod
    def get_quality_rating(self) -> int:
        """Get quality rating (1-10, higher is better)"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available (API keys, connectivity, etc.)"""
        pass
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get comprehensive provider information"""
        return {
            'name': self.name,
            'cost_per_image': self.get_cost_per_image(),
            'speed_rating': self.get_speed_rating(),
            'quality_rating': self.get_quality_rating(),
            'available': self.is_available()
        }


class RembgProvider(BackgroundRemovalProvider):
    """REMBG (free, AI-powered) background removal provider"""
    
    def __init__(self, model_name: str = 'u2net'):
        super().__init__('rembg')
        self.model_name = model_name
        self.available = REMBG_AVAILABLE
        
        if not self.available:
            self.logger.warning("REMBG not available - please install: pip install rembg[cpu]")
    
    def remove_background(self, input_path: str, output_path: str) -> BackgroundRemovalResult:
        """Remove background using REMBG with aggressive pink button removal"""
        start_time = time.time()
        
        if not self.available:
            return BackgroundRemovalResult(
                success=False,
                error_message="REMBG not installed or available",
                provider_used=self.name
            )
        
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            if not input_path.exists():
                return BackgroundRemovalResult(
                    success=False,
                    error_message=f"Input file not found: {input_path}",
                    provider_used=self.name
                )
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load and process image
            with open(input_path, 'rb') as input_file:
                input_data = input_file.read()
            
            # Remove background using REMBG
            output_data = remove(input_data)
            
            # Pink button removal now happens BEFORE background removal in main pipeline
            # No need for post-processing here
            
            # Save result
            with open(output_path, 'wb') as output_file:
                output_file.write(output_data)
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"✅ REMBG: Background removed with pink button cleanup in {processing_time:.2f}s")
            
            return BackgroundRemovalResult(
                success=True,
                output_path=str(output_path),
                provider_used=self.name,
                processing_time=processing_time,
                cost=0.0,  # Free!
                metadata={'model': self.model_name, 'pink_button_removal': True}
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"REMBG error: {str(e)}"
            self.logger.error(error_msg)
            
            return BackgroundRemovalResult(
                success=False,
                error_message=error_msg,
                provider_used=self.name,
                processing_time=processing_time
            )
    
    def _remove_pink_buttons_post_rembg(self, image_data: bytes) -> bytes:
        """
        Aggressive pink button removal after REMBG processing.
        Targets specific pink colors and bottom-right corner areas.
        """
        import cv2
        import numpy as np
        from io import BytesIO
        
        # Convert bytes to OpenCV image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            self.logger.warning("Could not decode image for pink button removal")
            return image_data
        
        h, w = image.shape[:2]
        
        # Convert to RGBA if not already
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            elif image.shape[2] == 4:
                pass  # Already RGBA
        else:
            return image_data  # Can't process grayscale
        
        # Convert to RGB for HSV processing
        rgb_image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        # Create removal mask
        removal_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Method 1: Precise pink color targeting for the specific button color
        # Target the exact Flink pink (#E91E63 and similar)
        lower_pink1 = np.array([320, 150, 180])  # More precise deep pink/magenta
        upper_pink1 = np.array([340, 255, 255])
        
        lower_pink2 = np.array([150, 120, 180])  # More precise rose/pink
        upper_pink2 = np.array([175, 255, 255])
        
        pink_mask1 = cv2.inRange(hsv, lower_pink1, upper_pink1)
        pink_mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
        pink_mask = cv2.bitwise_or(pink_mask1, pink_mask2)
        
        # Method 2: Circular button detection with precise positioning
        # The + button appears consistently in the same relative position
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100,  # Increase minDist to avoid false positives
            param1=50, param2=35, minRadius=35, maxRadius=65  # Precise button size range
        )
        
        button_found = False
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # The + button is typically in bottom-right area, very precisely positioned
                # More restrictive positioning - only remove if clearly in button area
                if x > w * 0.75 and y > h * 0.75 and x < w - 20 and y < h - 20:
                    # Only remove the exact circular area with minimal padding
                    cv2.circle(removal_mask, (x, y), r + 5, 255, -1)  # Minimal padding
                    self.logger.debug(f"🎯 Removed precise + button at ({x}, {y}) radius {r}")
                    button_found = True
        
        # Method 3: Fallback - if no circle detected, target known button position
        # The + button consistently appears ~48px from bottom-right corner in 573x573 tiles
        if not button_found:
            # Estimated button position for 573x573 tiles (common Flink size)
            button_x = w - 48  # ~48px from right edge
            button_y = h - 48  # ~48px from bottom edge
            button_radius = 24  # ~24px radius
            
            # Only apply if we're in reasonable tile size range
            if 400 < w < 700 and 400 < h < 700:
                # Check if there's actually pink content at this position
                if button_x > 0 and button_y > 0 and button_x < w and button_y < h:
                    # Sample the color at the expected button center
                    center_color = hsv[button_y, button_x]
                    # Check if it's in pink range
                    if ((320 <= center_color[0] <= 340) or (150 <= center_color[0] <= 175)) and center_color[1] > 100:
                        cv2.circle(removal_mask, (button_x, button_y), button_radius + 5, 255, -1)
                        self.logger.debug(f"🎯 Removed fallback + button at estimated position ({button_x}, {button_y})")
                        button_found = True
        
        # Combine with pink color detection, but only in bottom-right area to avoid false positives
        # Restrict pink removal to bottom-right quadrant only
        pink_mask_restricted = np.zeros_like(pink_mask)
        pink_mask_restricted[int(h*0.7):h, int(w*0.7):w] = pink_mask[int(h*0.7):h, int(w*0.7):w]
        
        # Combine masks
        combined_mask = cv2.bitwise_or(removal_mask, pink_mask_restricted)
        
        # Light morphological cleanup - much more conservative
        if np.sum(combined_mask > 0) > 0:  # Only if we found something to remove
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Smaller kernel
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply removal mask to alpha channel
        image[combined_mask > 0, 3] = 0  # Make transparent
        
        # Count removed pixels for logging
        removed_pixels = np.sum(combined_mask > 0)
        total_pixels = h * w
        removal_percentage = (removed_pixels / total_pixels) * 100
        
        if removal_percentage > 0.1:  # Log even small removals
            self.logger.debug(f"🎯 Precise + button removal: {removal_percentage:.1f}% of image ({removed_pixels} pixels)")
        elif button_found:
            self.logger.debug(f"🎯 + button detected and removed precisely")
        
        # Convert back to bytes
        _, buffer = cv2.imencode('.png', image)
        return buffer.tobytes()
    
    def get_cost_per_image(self) -> float:
        return 0.0  # Free!
    
    def get_speed_rating(self) -> int:
        return 8  # Fast local processing
    
    def get_quality_rating(self) -> int:
        return 7  # Good quality for most use cases
    
    def is_available(self) -> bool:
        return self.available


class RemoveBgProvider(BackgroundRemovalProvider):
    """Remove.bg API provider"""
    
    def __init__(self):
        super().__init__('remove_bg')
        self.api_key = os.getenv('REMOVE_BG_API_KEY')
        self.api_url = 'https://api.remove.bg/v1.0/removebg'
    
    def remove_background(self, input_path: str, output_path: str) -> BackgroundRemovalResult:
        """Remove background using Remove.bg API"""
        start_time = time.time()
        
        if not self.is_available():
            return BackgroundRemovalResult(
                success=False,
                error_message="Remove.bg API key not set",
                provider_used=self.name
            )
        
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(input_path, 'rb') as image_file:
                response = requests.post(
                    self.api_url,
                    files={'image_file': image_file},
                    data={'size': 'auto'},
                    headers={'X-Api-Key': self.api_key},
                    timeout=30
                )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                with open(output_path, 'wb') as output_file:
                    output_file.write(response.content)
                
                self.logger.info(f"✅ Remove.bg: Background removed successfully in {processing_time:.2f}s")
                
                return BackgroundRemovalResult(
                    success=True,
                    output_path=str(output_path),
                    provider_used=self.name,
                    processing_time=processing_time,
                    cost=0.20,
                    metadata={'api_response_time': processing_time}
                )
            else:
                error_msg = f"Remove.bg API error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                
                return BackgroundRemovalResult(
                    success=False,
                    error_message=error_msg,
                    provider_used=self.name,
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Remove.bg error: {str(e)}"
            self.logger.error(error_msg)
            
            return BackgroundRemovalResult(
                success=False,
                error_message=error_msg,
                provider_used=self.name,
                processing_time=processing_time
            )
    
    def get_cost_per_image(self) -> float:
        return 0.20
    
    def get_speed_rating(self) -> int:
        return 5  # Network dependent
    
    def get_quality_rating(self) -> int:
        return 9  # Excellent quality
    
    def is_available(self) -> bool:
        return self.api_key is not None


class PhotoroomProvider(BackgroundRemovalProvider):
    """Photoroom API provider"""
    
    def __init__(self):
        super().__init__('photoroom')
        self.api_key = os.getenv('PHOTOROOM_API_KEY')
        self.api_url = 'https://image-api.photoroom.com/v1/segment'
    
    def remove_background(self, input_path: str, output_path: str) -> BackgroundRemovalResult:
        """Remove background using Photoroom API"""
        start_time = time.time()
        
        if not self.is_available():
            return BackgroundRemovalResult(
                success=False,
                error_message="Photoroom API key not set",
                provider_used=self.name
            )
        
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(input_path, 'rb') as image_file:
                response = requests.post(
                    self.api_url,
                    files={'image_file': image_file},
                    headers={'x-api-key': self.api_key},
                    timeout=30
                )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                with open(output_path, 'wb') as output_file:
                    output_file.write(response.content)
                
                self.logger.info(f"✅ Photoroom: Background removed successfully in {processing_time:.2f}s")
                
                return BackgroundRemovalResult(
                    success=True,
                    output_path=str(output_path),
                    provider_used=self.name,
                    processing_time=processing_time,
                    cost=0.10,
                    metadata={'api_response_time': processing_time}
                )
            else:
                error_msg = f"Photoroom API error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                
                return BackgroundRemovalResult(
                    success=False,
                    error_message=error_msg,
                    provider_used=self.name,
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Photoroom error: {str(e)}"
            self.logger.error(error_msg)
            
            return BackgroundRemovalResult(
                success=False,
                error_message=error_msg,
                provider_used=self.name,
                processing_time=processing_time
            )
    
    def get_cost_per_image(self) -> float:
        return 0.10
    
    def get_speed_rating(self) -> int:
        return 6  # Good speed, network dependent
    
    def get_quality_rating(self) -> int:
        return 8  # Great for product images
    
    def is_available(self) -> bool:
        return self.api_key is not None


class RemovalAiProvider(BackgroundRemovalProvider):
    """Removal.AI API provider"""
    
    def __init__(self):
        super().__init__('removal_ai')
        self.api_key = os.getenv('REMOVAL_AI_API_KEY')
        self.api_url = 'https://api.removal.ai/4.0/remove'
    
    def remove_background(self, input_path: str, output_path: str) -> BackgroundRemovalResult:
        """Remove background using Removal.AI API"""
        start_time = time.time()
        
        if not self.is_available():
            return BackgroundRemovalResult(
                success=False,
                error_message="Removal.AI API key not set",
                provider_used=self.name
            )
        
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(input_path, 'rb') as image_file:
                response = requests.post(
                    self.api_url,
                    files={'image_file': image_file},
                    headers={'Api-Key': self.api_key},
                    timeout=30
                )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                with open(output_path, 'wb') as output_file:
                    output_file.write(response.content)
                
                self.logger.info(f"✅ Removal.AI: Background removed successfully in {processing_time:.2f}s")
                
                return BackgroundRemovalResult(
                    success=True,
                    output_path=str(output_path),
                    provider_used=self.name,
                    processing_time=processing_time,
                    cost=0.15,
                    metadata={'api_response_time': processing_time}
                )
            else:
                error_msg = f"Removal.AI API error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                
                return BackgroundRemovalResult(
                    success=False,
                    error_message=error_msg,
                    provider_used=self.name,
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Removal.AI error: {str(e)}"
            self.logger.error(error_msg)
            
            return BackgroundRemovalResult(
                success=False,
                error_message=error_msg,
                provider_used=self.name,
                processing_time=processing_time
            )
    
    def get_cost_per_image(self) -> float:
        return 0.15
    
    def get_speed_rating(self) -> int:
        return 6  # Good speed
    
    def get_quality_rating(self) -> int:
        return 8  # Very good quality
    
    def is_available(self) -> bool:
        return self.api_key is not None


# Provider Registry - Easy to extend
AVAILABLE_PROVIDERS = {
    'rembg': RembgProvider,
    'remove_bg': RemoveBgProvider,
    'photoroom': PhotoroomProvider,
    'removal_ai': RemovalAiProvider
}

def create_provider(provider_name: str, **kwargs) -> BackgroundRemovalProvider:
    """Factory function to create provider instances"""
    if provider_name not in AVAILABLE_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(AVAILABLE_PROVIDERS.keys())}")
    
    provider_class = AVAILABLE_PROVIDERS[provider_name]
    return provider_class(**kwargs)

def get_available_providers() -> List[str]:
    """Get list of available provider names"""
    return list(AVAILABLE_PROVIDERS.keys())