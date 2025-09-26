#!/usr/bin/env python3
"""
Step 3: Product Canvas Detection
Detects product grid canvases within the content region using computer vision
"""

import sys
import cv2
import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project paths for imports
sys.path.append('src')
sys.path.append('.')

from src.interfaces import StepInput, StepOutput, CanvasDetectionResult
from src.screenshot_ui_analyzer import ScreenshotUIAnalyzer

# TileDetector class (inline implementation from step_by_step_pipeline.py)
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
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'area': area, 'aspect_ratio': aspect_ratio
                })

        return tile_candidates

    def _sort_tiles_grid(self, tiles: List[Tuple[int, int, int, int]], image_shape) -> List[Tuple[int, int, int, int]]:
        """Sort tiles in grid order (left to right, top to bottom)"""
        if not tiles:
            return tiles

        # Sort by Y coordinate first (top to bottom), then by X (left to right)
        sorted_tiles = sorted(tiles, key=lambda t: (t[1], t[0]))
        return sorted_tiles

def run(input_data: StepInput) -> StepOutput:
    """
    Detect product canvases in the content region

    Args:
        input_data: StepInput with image and UI region data

    Returns:
        StepOutput with detected canvases and grid information
    """
    try:
        print("🎯 STEP 3: Product Canvas Detection (573x813px each)")

        image = input_data.image
        image_name = input_data.image_name
        output_dir = input_data.current_image_dir

        if output_dir:
            output_dir.mkdir(exist_ok=True)

        # Get UI analysis from previous steps
        ui_analysis = input_data.data.get("ui_analysis", {})

        # First perform UI analysis to get regions if not provided
        if 'regions' not in ui_analysis:
            print("   🔍 Performing UI analysis to detect regions...")
            ui_analyzer = ScreenshotUIAnalyzer()
            analysis = ui_analyzer.analyze_screenshot(image)
            ui_regions = analysis.get('regions', {})
        else:
            ui_regions = ui_analysis['regions']

        # Get content region from Step 1 data
        content_region = input_data.data.get("content_region")

        if content_region:
            content_x = content_region['x']
            content_y = content_region['y']
            content_w = content_region['width']
            content_h = content_region['height']
        else:
            # Fallback: use detected content region or compute from image
            h, w = image.shape[:2]
            content_region = {
                'x': 0,
                'y': 531,  # Skip header region
                'width': w,
                'height': h - 531 - 250  # Skip footer region
            }
            content_x = content_region['x']
            content_y = content_region['y']
            content_w = content_region['width']
            content_h = content_region['height']
            print(f"   ⚠️  No content region from Step 1, using fallback: {content_w}x{content_h} at ({content_x}, {content_y})")

        print(f"   📐 Content region: {content_w}x{content_h} at ({content_x}, {content_y})")

        # Extract content area and detect tiles
        content_image = image[content_y:content_y+content_h, content_x:content_x+content_w]
        tile_detector = TileDetector()
        detected_tiles = tile_detector.detect_tiles(content_image)

        print(f"   🔍 Found {len(detected_tiles)} potential tiles")

        # Convert tiles to canvas rectangles (tile + text area = 573x813px)
        canvases = []
        for i, (tile_x, tile_y, tile_w, tile_h) in enumerate(detected_tiles):
            # Convert relative coordinates to absolute image coordinates
            abs_x = content_x + tile_x
            abs_y = content_y + tile_y

            canvas = {
                'canvas_id': i + 1,
                'x': abs_x,
                'y': abs_y,
                'width': 573,  # Standard canvas width
                'height': 813,  # Standard canvas height (573px tile + 240px text)
                'tile_region': {
                    'x': abs_x,
                    'y': abs_y,
                    'width': tile_w,
                    'height': tile_h
                },
                'text_region': {
                    'x': abs_x,
                    'y': abs_y + tile_h,
                    'width': 573,
                    'height': 240
                }
            }
            canvases.append(canvas)

        # Add canvas boundary validation to ensure they stay within content bounds
        valid_canvases = []
        for canvas in canvases:
            # Check if canvas fits within content region
            canvas_right = canvas['x'] + canvas['width']
            canvas_bottom = canvas['y'] + canvas['height']
            content_right = content_x + content_w
            content_bottom = content_y + content_h

            if (canvas['x'] >= content_x and canvas['y'] >= content_y and
                canvas_right <= content_right and canvas_bottom <= content_bottom):
                valid_canvases.append(canvas)
            else:
                print(f"   ⚠️  Canvas {canvas['canvas_id']} extends beyond content bounds, adjusting...")
                # Adjust canvas to fit within content bounds
                adjusted_canvas = canvas.copy()
                if canvas_right > content_right:
                    adjusted_canvas['width'] = content_right - canvas['x']
                if canvas_bottom > content_bottom:
                    adjusted_canvas['height'] = content_bottom - canvas['y']
                valid_canvases.append(adjusted_canvas)

        # Load category data from previous steps for CSV generation
        category_data = input_data.data.get("category_data", {})
        category = category_data.get('main_category', 'Unknown')
        subcategory = category_data.get('active_subcategory', 'Unknown')

        # Generate CSV data for each product canvas
        csv_data = []
        for canvas in valid_canvases:
            csv_row = {
                'product_canvas_id': f"{image_name}_canvas_{canvas['canvas_id']}",
                'category': category,
                'subcategory': subcategory,
                'canvas_x': canvas['x'],
                'canvas_y': canvas['y'],
                'canvas_width': canvas['width'],
                'canvas_height': canvas['height'],
                'source_image': image_name,
                'detection_confidence': 0.95,  # High confidence for tile-based detection
                'tile_x': canvas['tile_region']['x'],
                'tile_y': canvas['tile_region']['y'],
                'tile_width': canvas['tile_region']['width'],
                'tile_height': canvas['tile_region']['height']
            }
            csv_data.append(csv_row)

        output_files = {}
        if output_dir:
            # Create visualization
            vis_image = image.copy()

            # Draw content region outline
            cv2.rectangle(vis_image, (content_x, content_y), (content_x + content_w, content_y + content_h), (0, 255, 0), 2)
            cv2.putText(vis_image, "CONTENT REGION", (content_x + 10, content_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw each canvas with proper 573x813 dimensions
            for canvas in valid_canvases:
                x, y, w, h = canvas['x'], canvas['y'], canvas['width'], canvas['height']
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cv2.putText(vis_image, f"Canvas {canvas['canvas_id']} ({w}x{h})", (x + 10, y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Draw tile region within canvas
                tile = canvas['tile_region']
                cv2.rectangle(vis_image, (tile['x'], tile['y']),
                             (tile['x'] + tile['width'], tile['y'] + tile['height']), (0, 255, 255), 2)
                cv2.putText(vis_image, "PRODUCT", (tile['x'] + 10, tile['y'] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Save visualization
            canvas_path = output_dir / f"{image_name}_02_canvases.jpg"
            cv2.imwrite(str(canvas_path), vis_image)
            output_files["canvas_visualization"] = str(canvas_path)

            # Save CSV file with product canvas data
            csv_path = output_dir / f"{image_name}_02_product_canvases.csv"
            if csv_data:
                df = pd.DataFrame(csv_data)
                df.to_csv(csv_path, index=False)
                print(f"  📊 Saved CSV with {len(csv_data)} product canvas rows: {csv_path}")
                output_files["canvas_csv"] = str(csv_path)
            else:
                print("  ⚠️  No product canvases found - CSV not created")

            # Save JSON data
            result_data = {
                "step": "02_product_canvas_detection",
                "description": f"Detected {len(valid_canvases)} product canvases (573x813px each)",
                "content_region": content_region,
                "canvases": valid_canvases,
                "category_data": category_data,
                "csv_data": csv_data
            }

            json_path = output_dir / f"{image_name}_02_canvases.json"
            with open(json_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            output_files["canvas_data"] = str(json_path)

        print(f"  ✅ Created {len(valid_canvases)} product canvases (573x813px each)")
        if output_dir:
            print(f"  ✅ Saved: {canvas_path}")
            print(f"  ✅ Saved: {json_path}")

        return StepOutput(
            success=True,
            step_name="Canvas Detection",
            data={
                "canvases": valid_canvases,
                "content_region": content_region,
                "csv_data": csv_data,
                "category_data": category_data
            },
            output_files=output_files
        )

    except Exception as e:
        print(f"   ❌ Canvas detection failed: {e}")
        return StepOutput(
            success=False,
            step_name="Canvas Detection",
            errors=[f"Canvas detection failed: {str(e)}"]
        )