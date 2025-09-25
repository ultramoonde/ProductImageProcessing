import cv2
import numpy as np
from typing import List, Tuple, Optional

class TileDetector:
    def __init__(self, tile_size: int = 191, grid_cols: int = 4):
        self.tile_size = tile_size
        self.grid_cols = grid_cols
        
    def detect_tiles(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect product tiles using sophisticated HSV + grayscale detection.
        Returns list of (x, y, width, height) tuples for each detected tile.
        """
        # Use the proven sophisticated detection method
        tile_candidates = self.detect_gray_tiles_sophisticated(image)
        
        # Convert to the expected format
        tiles = []
        for tile in tile_candidates:
            tiles.append((tile['x'], tile['y'], tile['w'], tile['h']))
        
        return self._sort_tiles_grid(tiles, image.shape)
    
    def detect_gray_tiles_sophisticated(self, image: np.ndarray) -> list:
        """Detect light gray product tiles using the proven working method from ai_enhanced_extraction."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Combined HSV and grayscale detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # HSV range for light gray regions (proven working values)
        lower_tile = np.array([0, 0, 230])
        upper_tile = np.array([180, 30, 255])
        
        hsv_mask = cv2.inRange(hsv, lower_tile, upper_tile)
        
        # Remove very white areas using grayscale
        gray_only_mask = cv2.inRange(gray, 235, 250)
        
        # Combine masks (use the working version)
        tile_mask = cv2.bitwise_and(hsv_mask, gray_only_mask)
        
        # Clean up the mask
        kernel = np.ones((5,5), np.uint8)
        tile_mask = cv2.morphologyEx(tile_mask, cv2.MORPH_CLOSE, kernel)
        tile_mask = cv2.morphologyEx(tile_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(tile_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours for valid tiles (using proven working criteria)
        tile_candidates = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Filter for reasonable tile sizes (proven working criteria)
            if (area > 150000 and
                0.8 <= aspect_ratio <= 1.25 and
                w > 400 and h > 400):
                tile_candidates.append({
                    'index': i,
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'area': area,
                    'aspect_ratio': aspect_ratio
                })
        
        # Sort by area and take up to 4 (Flink shows 4 products per viewport)
        tile_candidates.sort(key=lambda t: t['area'], reverse=True)
        return tile_candidates[:4]
    
    def detect_tiles_template_matching(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Alternative method using template matching for more precise detection.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create a template for rounded rectangle detection
        template = self._create_rounded_rect_template()
        
        # Apply template matching
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        
        # Find locations with high correlation
        threshold = 0.6
        locations = np.where(result >= threshold)
        
        tiles = []
        for pt in zip(*locations[::-1]):  # Switch x and y
            x, y = pt
            tiles.append((x, y, self.tile_size, self.tile_size))
        
        # Remove overlapping detections
        tiles = self._remove_overlapping_tiles(tiles)
        
        return self._sort_tiles_grid(tiles, image.shape)
    
    def _create_rounded_rect_template(self) -> np.ndarray:
        """Create a template of a rounded rectangle for matching."""
        template = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
        
        # Create rounded rectangle
        radius = 16
        cv2.rectangle(template, (radius, 0), (self.tile_size - radius, self.tile_size), 255, -1)
        cv2.rectangle(template, (0, radius), (self.tile_size, self.tile_size - radius), 255, -1)
        
        # Add rounded corners
        cv2.circle(template, (radius, radius), radius, 255, -1)
        cv2.circle(template, (self.tile_size - radius, radius), radius, 255, -1)
        cv2.circle(template, (radius, self.tile_size - radius), radius, 255, -1)
        cv2.circle(template, (self.tile_size - radius, self.tile_size - radius), radius, 255, -1)
        
        return template
    
    def _remove_overlapping_tiles(self, tiles: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Remove overlapping tile detections."""
        if not tiles:
            return tiles
        
        # Sort by confidence or area (for now, just keep first occurrence)
        unique_tiles = []
        for tile in tiles:
            x1, y1, w1, h1 = tile
            is_duplicate = False
            
            for existing in unique_tiles:
                x2, y2, w2, h2 = existing
                
                # Check if tiles overlap significantly
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                if overlap_area > (w1 * h1 * 0.5):  # 50% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_tiles.append(tile)
        
        return unique_tiles
    
    def _sort_tiles_grid(self, tiles: List[Tuple[int, int, int, int]], image_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int, int]]:
        """Sort tiles in grid order (left to right, top to bottom)."""
        if not tiles:
            return tiles
        
        # Sort by y coordinate first (top to bottom), then by x coordinate (left to right)
        tiles.sort(key=lambda tile: (tile[1], tile[0]))
        
        return tiles
    
    def visualize_detections(self, image: np.ndarray, tiles: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Draw detected tiles on image for debugging."""
        result = image.copy()
        
        for i, (x, y, w, h) in enumerate(tiles):
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, f'Tile {i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return result