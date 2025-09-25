import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple, List, Optional
import logging
from pathlib import Path

# Import the new background removal system
try:
    from background_removal_manager import BackgroundRemovalManager
    BACKGROUND_REMOVAL_AVAILABLE = True
except ImportError:
    BACKGROUND_REMOVAL_AVAILABLE = False
    print("Warning: Background removal system not available")

class ImageProcessor:
    def __init__(self, tile_size: int = 191, corner_radius: int = 16, 
                 enable_background_removal: bool = True, bg_config_path: Optional[str] = None):
        self.tile_size = tile_size
        self.corner_radius = corner_radius
        self.enable_background_removal = enable_background_removal
        
        # Initialize background removal manager
        self.bg_manager = None
        if enable_background_removal and BACKGROUND_REMOVAL_AVAILABLE:
            try:
                config_path = bg_config_path or Path(__file__).parent.parent / "bg_removal_config.yaml"
                self.bg_manager = BackgroundRemovalManager(str(config_path) if config_path.exists() else None)
                print(f"✅ Background removal system initialized")
            except Exception as e:
                print(f"⚠️  Background removal initialization failed: {str(e)}")
                self.bg_manager = None
        elif enable_background_removal:
            print("⚠️  Background removal requested but system not available")
        
        self.logger = logging.getLogger('ImageProcessor')
    
    def extract_tile_with_rounded_corners(self, image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Extract a tile from the image and apply rounded corners with transparency.
        """
        # Extract the tile region
        tile = image[y:y+height, x:x+width]
        
        # Resize to exact tile size if needed
        if tile.shape[:2] != (self.tile_size, self.tile_size):
            tile = cv2.resize(tile, (self.tile_size, self.tile_size))
        
        # Convert to PIL for easier transparency handling
        pil_image = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        
        # Create rounded corner mask
        mask = self._create_rounded_mask(self.tile_size, self.tile_size, self.corner_radius)
        
        # Apply mask to create transparency
        pil_image.putalpha(mask)
        
        # Convert back to numpy array (RGBA)
        result = np.array(pil_image)
        
        return result
    
    def process_extracted_tile(self, raw_tile: np.ndarray) -> np.ndarray:
        """
        Process a raw extracted tile (resize and add rounded corners).
        This is used after pink button removal has been done on the full-size tile.
        """
        # Resize to exact tile size
        if raw_tile.shape[:2] != (self.tile_size, self.tile_size):
            tile = cv2.resize(raw_tile, (self.tile_size, self.tile_size))
        else:
            tile = raw_tile.copy()
        
        # Convert to PIL for easier transparency handling
        pil_image = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        
        # Create rounded corner mask
        mask = self._create_rounded_mask(self.tile_size, self.tile_size, self.corner_radius)
        
        # Apply mask to create transparency
        pil_image.putalpha(mask)
        
        # Convert back to numpy array (RGBA)
        result = np.array(pil_image)
        
        return result
    
    def _create_rounded_mask(self, width: int, height: int, radius: int) -> Image.Image:
        """Create a mask for rounded corners."""
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Draw rounded rectangle
        draw.rounded_rectangle([(0, 0), (width, height)], radius=radius, fill=255)
        
        return mask
    
    def process_tile(self, tile_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a tile image and return processed tile and text region.
        Returns: (processed_tile, text_region) 
        """
        h, w = tile_image.shape[:2]
        channels = tile_image.shape[2] if len(tile_image.shape) > 2 else 1
        input_size_mb = (h * w * channels * tile_image.dtype.itemsize) / (1024 * 1024)
        
        print(f"🔄 PROCESS TILE PIPELINE START:")
        print(f"   📥 Input tile dimensions: {w}x{h} (channels: {channels})")
        print(f"   💾 Input size: {input_size_mb:.2f}MB")
        print(f"   📊 Input data type: {tile_image.dtype}")
        print(f"   🎯 Target output size: {self.tile_size}x{self.tile_size}")
        
        # Step 1: Remove pink button using fixed coordinate-based approach
        print(f"   🔸 STEP 1: Pink button removal...")
        clean_tile = self.remove_pink_button_fixed_position(tile_image)
        
        clean_h, clean_w = clean_tile.shape[:2]
        clean_channels = clean_tile.shape[2] if len(clean_tile.shape) > 2 else 1
        clean_size_mb = (clean_h * clean_w * clean_channels * clean_tile.dtype.itemsize) / (1024 * 1024)
        
        print(f"   ✅ Step 1 complete - Clean tile: {clean_w}x{clean_h} ({clean_size_mb:.2f}MB)")
        
        # Step 2: Process the clean tile (resize, round corners, etc.)
        print(f"   🔸 STEP 2: Tile processing (resize + round corners)...")
        processed_tile = self.process_extracted_tile(clean_tile)
        
        proc_h, proc_w = processed_tile.shape[:2]
        proc_channels = processed_tile.shape[2] if len(processed_tile.shape) > 2 else 1
        proc_size_mb = (proc_h * proc_w * proc_channels * processed_tile.dtype.itemsize) / (1024 * 1024)
        
        print(f"   ✅ Step 2 complete - Processed tile: {proc_w}x{proc_h} ({proc_size_mb:.2f}MB)")
        print(f"   📐 Size reduction: {input_size_mb:.2f}MB → {proc_size_mb:.2f}MB ({((proc_size_mb/input_size_mb)*100):.1f}%)")
        
        # Step 3: Create text region (OCR-optimized version)
        print(f"   🔸 STEP 3: Text region preparation...")
        # For text region, we'll use the processed tile for now
        # In the future, this could be a separate OCR-optimized version
        text_region = processed_tile.copy()
        
        text_h, text_w = text_region.shape[:2]
        text_channels = text_region.shape[2] if len(text_region.shape) > 2 else 1
        
        print(f"   ✅ Step 3 complete - Text region: {text_w}x{text_h} (channels: {text_channels})")
        
        print(f"🎯 PROCESS TILE PIPELINE COMPLETE:")
        print(f"   📤 Final processed tile: {proc_w}x{proc_h} (RGBA: {proc_channels==4})")
        print(f"   📤 Final text region: {text_w}x{text_h} (RGBA: {text_channels==4})")
        print(f"   ⚡ Pipeline efficiency: {((proc_size_mb/input_size_mb)*100):.1f}% size retention")
        
        return processed_tile, text_region
    
    def preprocess_for_ocr(self, tile: np.ndarray) -> np.ndarray:
        """
        Preprocess tile image for better OCR results.
        """
        # Convert to grayscale if it's color
        if len(tile.shape) == 3:
            if tile.shape[2] == 4:  # RGBA
                # Convert RGBA to RGB first
                rgb = cv2.cvtColor(tile, cv2.COLOR_RGBA2RGB)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            else:  # RGB or BGR
                gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        else:
            gray = tile.copy()
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(denoised)
        
        # Apply slight Gaussian blur to smooth text
        smoothed = cv2.GaussianBlur(contrast_enhanced, (1, 1), 0)
        
        return smoothed
    
    def save_tile_as_png(self, tile: np.ndarray, filepath: str, remove_background: bool = None) -> bool:
        """
        Save tile as PNG with optional background removal and transparency preserved.
        
        Args:
            tile: Image tile as numpy array
            filepath: Output file path
            remove_background: Override default background removal setting
            
        Returns:
            bool: Success status
        """
        try:
            filepath = Path(filepath)
            h, w = tile.shape[:2]
            channels = tile.shape[2] if len(tile.shape) > 2 else 1
            
            # Determine if we should remove background
            should_remove_bg = (
                remove_background if remove_background is not None 
                else self.enable_background_removal
            )
            
            print(f"🖼️  SAVE TILE DEBUG:")
            print(f"   📁 Output path: {filepath}")
            print(f"   📐 Tile dimensions: {w}x{h} (channels: {channels})")
            print(f"   🔧 Background removal requested: {remove_background}")
            print(f"   🔧 Default background removal: {self.enable_background_removal}")
            print(f"   🎯 Will remove background: {should_remove_bg}")
            print(f"   🤖 Background manager available: {self.bg_manager is not None}")
            
            # First, save the original tile
            temp_path = None
            if should_remove_bg and self.bg_manager:
                # Save to temporary file first
                temp_path = filepath.parent / f"temp_{filepath.name}"
                success = self._save_raw_tile(tile, str(temp_path))
                
                if success:
                    # Use background removal manager to process
                    result = self.bg_manager.process_with_fallback(
                        str(temp_path), 
                        str(filepath)
                    )
                    
                    if result.success:
                        self.logger.info(f"Background removed using {result.provider_used} "
                                       f"(quality: {result.quality_score:.3f}, cost: ${result.cost:.3f})")
                        
                        # Clean up temp file
                        if temp_path.exists():
                            temp_path.unlink()
                        
                        return True
                    else:
                        self.logger.warning(f"Background removal failed: {result.error_message}")
                        # Fall back to saving original
                        if temp_path.exists():
                            temp_path.rename(filepath)
                        return True
                else:
                    self.logger.error(f"Failed to save temporary file for background removal")
                    return False
            else:
                # Save directly without background removal
                return self._save_raw_tile(tile, str(filepath))
                
        except Exception as e:
            self.logger.error(f"Error saving tile to {filepath}: {e}")
            # Clean up temp file if it exists
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            return False
    
    def _save_raw_tile(self, tile: np.ndarray, filepath: str) -> bool:
        """Save raw tile without background removal"""
        try:
            if len(tile.shape) == 3 and tile.shape[2] == 4:  # RGBA
                pil_image = Image.fromarray(tile, 'RGBA')
            else:  # RGB or grayscale
                if len(tile.shape) == 3:
                    pil_image = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
                else:
                    pil_image = Image.fromarray(tile, 'L')
            
            pil_image.save(filepath, 'PNG')
            return True
        except Exception as e:
            print(f"Error saving raw tile to {filepath}: {e}")
            return False
    
    def set_background_removal_strategy(self, strategy: str) -> bool:
        """
        Change background removal strategy dynamically
        
        Args:
            strategy: Strategy name ('cost_first', 'quality_first', 'speed_first', etc.)
            
        Returns:
            bool: Success status
        """
        if self.bg_manager:
            return self.bg_manager.set_strategy(strategy)
        return False
    
    def get_background_removal_stats(self) -> Optional[dict]:
        """Get background removal processing statistics"""
        if self.bg_manager:
            return self.bg_manager.get_processing_stats()
        return None
    
    def estimate_background_removal_cost(self, image_count: int) -> Optional[dict]:
        """Estimate cost for processing a batch of images"""
        if self.bg_manager:
            return self.bg_manager.estimate_cost(image_count)
        return None
    
    def extract_header_region(self, image: np.ndarray, header_height: int = 100) -> np.ndarray:
        """
        Extract the header region containing category information.
        """
        height = image.shape[0]
        header = image[0:header_height, :]
        return header
    
    def resize_with_aspect_ratio(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio, padding if necessary.
        """
        target_width, target_height = target_size
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(target_width / w, target_height / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        if len(image.shape) == 3:
            padded = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
        else:
            padded = np.zeros((target_height, target_width), dtype=image.dtype)
        
        # Center the resized image
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        
        if len(image.shape) == 3:
            padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        else:
            padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded
    
    def remove_pink_button_fixed_position(self, image: np.ndarray) -> np.ndarray:
        """
        DEPRECATED: Fixed coordinate approach - use detect_and_remove_pink_button instead.
        """
        print(f"⚠️  Using deprecated fixed coordinate method - switching to dynamic detection...")
        return self.detect_and_remove_pink_button(image)
    
    def detect_and_remove_pink_button(self, image: np.ndarray) -> np.ndarray:
        """
        Dynamically detect and remove pink circular button using computer vision.
        Uses HSV color detection + contour analysis to find the pink button.
        """
        result = image.copy()
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        print(f"🔍 DYNAMIC PINK BUTTON DETECTION:")
        print(f"   📐 Input image dimensions: {w}x{h} (channels: {channels})")
        
        if channels < 3:
            print(f"   ⚠️  Grayscale image - cannot detect pink button")
            return result
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        
        # Define pink/magenta color ranges in HSV
        # Pink button appears to be a bright magenta/pink color
        lower_pink1 = np.array([140, 100, 150])  # Lower pink range
        upper_pink1 = np.array([180, 255, 255])  # Upper pink range
        lower_pink2 = np.array([300, 100, 150])  # Alternate pink range (wrapping)
        upper_pink2 = np.array([360, 255, 255])  # Alternate pink range
        
        # Create masks for pink detection
        mask1 = cv2.inRange(hsv, lower_pink1, upper_pink1)
        mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
        pink_mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel)
        pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in the pink mask
        contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        buttons_found = 0
        buttons_removed = 0
        
        print(f"   🔍 Found {len(contours)} pink regions")
        
        for i, contour in enumerate(contours):
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Skip very small contours
            if area < 100:  # Minimum area threshold
                continue
                
            # Calculate circularity (4π*area/perimeter²)
            # Perfect circle = 1.0, square ≈ 0.785
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
                
            # Get bounding circle
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            center_x, center_y = int(center_x), int(center_y)
            radius = int(radius)
            
            # Calculate position relative to image
            pos_x_pct = (center_x / w) * 100
            pos_y_pct = (center_y / h) * 100
            
            print(f"   🔍 Pink region {i+1}:")
            print(f"      Area: {area:.0f}px², Radius: {radius}px")
            print(f"      Circularity: {circularity:.3f}")
            print(f"      Position: ({center_x}, {center_y}) = ({pos_x_pct:.1f}%, {pos_y_pct:.1f}%)")
            
            # Check if this looks like a button:
            # 1. Reasonably circular (> 0.6)
            # 2. Reasonable size (radius 10-50px)  
            # 3. Located in bottom or right area (common UI button positions)
            is_circular = circularity > 0.6
            is_good_size = 10 <= radius <= 50
            is_button_position = (pos_x_pct > 60 or pos_y_pct > 60)  # Bottom-right area or right edge
            
            if is_circular and is_good_size and is_button_position:
                buttons_found += 1
                print(f"   ✅ BUTTON DETECTED - Removing at ({center_x}, {center_y}) with radius {radius}")
                
                # Create removal mask - use slightly larger radius for complete removal
                removal_radius = radius + 5
                removal_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(removal_mask, (center_x, center_y), removal_radius, 255, -1)
                
                # Fill with background color that matches the tile
                # Use color from nearby non-button area
                bg_color = self._get_background_color(result, center_x, center_y, removal_radius + 10)
                result[removal_mask > 0] = bg_color
                
                buttons_removed += 1
                
                # Count pixels modified
                pixels_modified = np.sum(removal_mask > 0)
                total_pixels = w * h
                percentage = (pixels_modified / total_pixels) * 100
                print(f"      Modified {pixels_modified:,} pixels ({percentage:.2f}%)")
            else:
                print(f"      ❌ Not a button: circular={is_circular}, size={is_good_size}, position={is_button_position}")
        
        print(f"   🎯 DETECTION SUMMARY:")
        print(f"      Pink regions found: {len(contours)}")
        print(f"      Buttons detected: {buttons_found}")
        print(f"      Buttons removed: {buttons_removed}")
        
        if buttons_removed == 0:
            print(f"   ⚠️  No buttons removed - image may already be clean")
        else:
            print(f"   ✅ Successfully removed {buttons_removed} pink button(s)")
        
        return result
    
    def _get_background_color(self, image: np.ndarray, center_x: int, center_y: int, avoid_radius: int) -> tuple:
        """Get representative background color avoiding the button area."""
        h, w = image.shape[:2]
        
        # Sample points around the button area but outside the avoid radius
        sample_points = [
            (max(0, center_x - avoid_radius - 20), center_y),  # Left
            (min(w-1, center_x + avoid_radius + 20), center_y),  # Right  
            (center_x, max(0, center_y - avoid_radius - 20)),  # Top
            (center_x, min(h-1, center_y + avoid_radius + 20)),  # Bottom
        ]
        
        colors = []
        for x, y in sample_points:
            if 0 <= x < w and 0 <= y < h:
                colors.append(image[y, x])
        
        if colors:
            # Return median color
            colors = np.array(colors)
            return tuple(np.median(colors, axis=0).astype(int))
        else:
            # Fallback to light gray
            return (242, 242, 242)
        
    def remove_pink_buttons_and_ui(self, image: np.ndarray) -> np.ndarray:
        """
        DEPRECATED: Old complex color-based method with incorrect HSV ranges.
        Use remove_pink_button_fixed_position() instead.
        """
        # For now, just call the new fixed position method
        return self.remove_pink_button_fixed_position(image)