#!/usr/bin/env python3
"""
Step 4: Component Extraction & Product Processing
Extracts individual product components and processes them with pink button removal
"""

import sys
import cv2
import numpy as np
import json
import pandas as pd
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add project paths for imports
sys.path.append('src')
sys.path.append('.')

from src.interfaces import StepInput, StepOutput, ComponentExtractionResult

def run(input_data: StepInput) -> StepOutput:
    """
    Extract individual product components and create clean product images

    Args:
        input_data: StepInput with image and canvas data

    Returns:
        StepOutput with component coordinates and clean product images
    """
    try:
        print("📍 STEP 4: Component Coordinate Extraction (Fixed)")

        image = input_data.image
        image_name = input_data.image_name
        output_dir = input_data.current_image_dir

        if output_dir:
            output_dir.mkdir(exist_ok=True)

        # Get canvases from previous step
        canvases = input_data.data.get("canvases", [])

        components_data = []

        for canvas in canvases:
            canvas_id = canvas['canvas_id']
            print(f"     🔍 Processing Canvas {canvas_id}")

            # Extract canvas region for analysis
            x, y, w, h = canvas['x'], canvas['y'], canvas['width'], canvas['height']

            # Ensure coordinates are within image bounds
            x = max(0, min(x, image.shape[1] - 1))
            y = max(0, min(y, image.shape[0] - 1))
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)

            canvas_region = image[y:y+h, x:x+w]

            # FIXED: Detect pink button with corrected HSV ranges
            pink_button = _detect_pink_button_in_tile_FIXED(canvas_region, x, y)

            # Prepare component data
            component = {
                'component_id': f"{image_name}_component_{canvas_id}",
                'canvas_id': canvas_id,
                'processing_timestamp': datetime.now().isoformat(),
                'canvas_x': x,
                'canvas_y': y,
                'canvas_width': w,
                'canvas_height': h,
                'product_image_x': x,
                'product_image_y': y,
                'product_image_width': canvas['tile_region']['width'],
                'product_image_height': canvas['tile_region']['height'],
                'text_area_x': canvas['text_region']['x'],
                'text_area_y': canvas['text_region']['y'],
                'text_area_width': canvas['text_region']['width'],
                'text_area_height': canvas['text_region']['height'],
                'pink_button_detected': pink_button['detected'],
                'pink_button_x': pink_button.get('x'),
                'pink_button_y': pink_button.get('y'),
                'pink_button_width': pink_button.get('w'),
                'pink_button_height': pink_button.get('h'),
                'pink_button_center_x': pink_button.get('center_x'),
                'pink_button_center_y': pink_button.get('center_y'),
                'pink_button_confidence': pink_button.get('confidence', 0)
            }

            components_data.append(component)

            if pink_button['detected']:
                print(f"       ✅ Pink button detected at ({pink_button['center_x']}, {pink_button['center_y']})")
            else:
                print(f"       ⚠️  No pink button detected")

        output_files = {}

        # Generate CSV file with component-level data (moved outside output_dir check)
        csv_data = []
        for comp in components_data:
            csv_row = {
                'component_id': f"{image_name}_component_{comp['canvas_id']}",
                'canvas_id': comp['canvas_id'],
                'processing_timestamp': comp['processing_timestamp'],
                'canvas_x': comp['canvas_x'],
                'canvas_y': comp['canvas_y'],
                'canvas_width': comp['canvas_width'],
                'canvas_height': comp['canvas_height'],
                'product_image_x': comp['product_image_x'],
                'product_image_y': comp['product_image_y'],
                'product_image_width': comp['product_image_width'],
                'product_image_height': comp['product_image_height'],
                'text_area_x': comp['text_area_x'],
                'text_area_y': comp['text_area_y'],
                'text_area_width': comp['text_area_width'],
                'text_area_height': comp['text_area_height'],
                'pink_button_detected': comp['pink_button_detected'],
                'pink_button_x': comp.get('pink_button_x', ''),
                'pink_button_y': comp.get('pink_button_y', ''),
                'pink_button_width': comp.get('pink_button_width', ''),
                'pink_button_height': comp.get('pink_button_height', ''),
                'pink_button_center_x': comp.get('pink_button_center_x', ''),
                'pink_button_center_y': comp.get('pink_button_center_y', ''),
                'pink_button_confidence': comp.get('pink_button_confidence', 0.0)
            }
            csv_data.append(csv_row)

        if output_dir:
            # Save components data
            with open(output_dir / f"{image_name}_components.json", 'w') as f:
                json.dump(components_data, f, indent=2)
            output_files["components_data"] = str(output_dir / f"{image_name}_components.json")

            # Save CSV file
            csv_path = output_dir / f"{image_name}_03_components.csv"
            if csv_data:
                df = pd.DataFrame(csv_data)
                df.to_csv(csv_path, index=False)
                print(f"  📊 Saved CSV with {len(csv_data)} component rows: {csv_path}")
                output_files["components_csv"] = str(csv_path)

            # Create enhanced visualization with ALL coordinate sections clearly marked
            vis_image = image.copy()
            for i, comp in enumerate(components_data):
                canvas_id = i + 1

                # 🟢 Green: Canvas boundary (overall product tile)
                cv2.rectangle(vis_image, (comp['canvas_x'], comp['canvas_y']),
                             (comp['canvas_x'] + comp['canvas_width'], comp['canvas_y'] + comp['canvas_height']),
                             (0, 255, 0), 3)
                cv2.putText(vis_image, f"CANVAS {canvas_id}",
                           (comp['canvas_x'] + 5, comp['canvas_y'] + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 🔵 Blue: Product image section within canvas
                cv2.rectangle(vis_image, (comp['product_image_x'], comp['product_image_y']),
                             (comp['product_image_x'] + comp['product_image_width'],
                              comp['product_image_y'] + comp['product_image_height']),
                             (255, 0, 0), 2)
                cv2.putText(vis_image, f"IMG {canvas_id}",
                           (comp['product_image_x'] + 5, comp['product_image_y'] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # 🟡 Yellow: Text area section within canvas
                cv2.rectangle(vis_image, (comp['text_area_x'], comp['text_area_y']),
                             (comp['text_area_x'] + comp['text_area_width'],
                              comp['text_area_y'] + comp['text_area_height']),
                             (0, 255, 255), 2)
                cv2.putText(vis_image, f"TXT {canvas_id}",
                           (comp['text_area_x'] + 5, comp['text_area_y'] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 🔴 Red: Pink button within product image (if detected)
                if comp['pink_button_detected']:
                    center_x = comp['pink_button_center_x']
                    center_y = comp['pink_button_center_y']
                    # Draw full circular button detection
                    cv2.circle(vis_image, (center_x, center_y), 48, (0, 0, 255), 3)
                    cv2.circle(vis_image, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.putText(vis_image, f"BTN {canvas_id}",
                               (center_x - 30, center_y - 55),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # Add confidence score
                    cv2.putText(vis_image, f"{comp['pink_button_confidence']:.1%}",
                               (center_x - 30, center_y + 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            vis_path = output_dir / f"{image_name}_03_components.jpg"
            cv2.imwrite(str(vis_path), vis_image)
            output_files["component_visualization"] = str(vis_path)

        print(f"   ✅ Extracted coordinates for {len(components_data)} canvases")
        if output_dir:
            print(f"   📁 Saved: {vis_path}")

        # NEW: Extract clean product images using coordinates
        print("🖼️ Extracting clean product images from component coordinates")
        clean_products = _extract_clean_product_images(image, components_data, image_name, output_dir)

        # Initialize background removal manager
        bg_removal_manager = _initialize_background_removal_manager()

        # STEP 4C: Apply background removal to hole-punched products (two-step process)
        for clean_product in clean_products:
            if bg_removal_manager and clean_product.get('holes_image_path'):
                try:
                    # Apply background removal to the hole-punched image
                    bg_result = _apply_background_removal(
                        clean_product['holes_image_path'],  # Use hole-punched image
                        bg_removal_manager,
                        output_dir
                    )
                    if bg_result['success']:
                        clean_product['bg_removed_path'] = bg_result['output_path']
                        clean_product['background_removed'] = True
                        if output_dir:
                            output_files[f"bg_removed_{clean_product['component_id']}"] = bg_result['output_path']
                    else:
                        clean_product['background_removed'] = False
                except Exception as e:
                    print(f"   ⚠️ Background removal failed for {clean_product['component_id']}: {e}")
                    clean_product['background_removed'] = False

        # Create compatibility layer: convert components_data to tiles format for backward compatibility
        tiles = []
        for comp in components_data:
            tile = {
                'x': comp['canvas_x'],
                'y': comp['canvas_y'],
                'w': comp['canvas_width'],
                'h': comp['canvas_height'],
                'canvas_id': comp['canvas_id'],
                'pink_button': {
                    'detected': comp['pink_button_detected'],
                    'x': comp.get('pink_button_x'),
                    'y': comp.get('pink_button_y'),
                    'center_x': comp.get('pink_button_center_x'),
                    'center_y': comp.get('pink_button_center_y'),
                    'confidence': comp.get('pink_button_confidence', 0)
                } if comp['pink_button_detected'] else {'detected': False}
            }
            tiles.append(tile)

        return StepOutput(
            success=True,
            step_name="Component Extraction",
            data={
                "components_data": components_data,
                "csv_data": csv_data,
                "clean_products": clean_products,  # NEW: Add clean products for Step 5
                "tiles": tiles  # For backward compatibility with subsequent steps
            },
            output_files=output_files
        )

    except Exception as e:
        print(f"   ❌ Component extraction failed: {e}")
        return StepOutput(
            success=False,
            step_name="Component Extraction",
            errors=[f"Component extraction failed: {str(e)}"]
        )


def _detect_pink_button_in_tile_FIXED(tile_region: np.ndarray, tile_x: int, tile_y: int) -> Dict[str, Any]:
    """
    PROVEN WORKING METHOD: Exact implementation from consolidated_pipeline.py
    This method successfully detects all 4 pink buttons with high accuracy
    """

    # Convert to HSV for better color detection (EXACT working method)
    hsv = cv2.cvtColor(tile_region, cv2.COLOR_BGR2HSV)

    # PROVEN WORKING parameters from Plan B Phase 1 (EXACT working method)
    lower_pink = np.array([140, 50, 50])    # PROVEN WORKING RANGE
    upper_pink = np.array([170, 255, 255])  # PROVEN WORKING RANGE

    # Create mask for pink color - EXACT WORKING RANGES
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Apply morphological operations to clean up the mask (EXACT working method)
    kernel = np.ones((5,5), np.uint8)  # 5x5 kernel as in working system
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours (EXACT working method)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # EXACT validation thresholds from working system
    min_area = 500           # Too small to be a button (working threshold)
    min_circularity = 0.7    # Only consider circular objects
    min_radius = 30          # Minimum button radius
    max_radius = 80          # Maximum button radius

    best_button = None
    best_score = 0

    for i, contour in enumerate(contours):
        # Calculate area and circularity (EXACT working validation)
        area = cv2.contourArea(contour)
        if area < min_area:  # EXACT working threshold
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Only consider circular objects (EXACT working validation)
        if circularity > min_circularity:
            # Get center and radius (EXACT working method)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center_x = int(x)
            center_y = int(y)
            radius = int(radius)

            # Additional validation: radius should be reasonable for buttons (EXACT working thresholds)
            if min_radius <= radius <= max_radius:
                # Convert to bounding box for compatibility with existing interface
                x_box = max(0, center_x - radius)
                y_box = max(0, center_y - radius)
                w_box = radius * 2
                h_box = radius * 2

                # Calculate confidence based on area and circularity (working method)
                confidence = min(area / 2000.0, 1.0) * circularity

                if confidence > best_score:
                    best_score = confidence
                    # Convert coordinates back to full image space
                    best_button = {
                        'detected': True,
                        'x': tile_x + x_box,
                        'y': tile_y + y_box,
                        'w': w_box,
                        'h': h_box,
                        'center_x': tile_x + center_x,
                        'center_y': tile_y + center_y,
                        'area': area,
                        'confidence': confidence,
                        'circularity': circularity,
                        'radius': radius
                    }

    if best_button is None:
        return {
            'detected': False,
            'x': None, 'y': None, 'w': None, 'h': None,
            'center_x': None, 'center_y': None,
            'area': 0, 'confidence': 0
        }

    return best_button


def _extract_clean_product_images(original_image: np.ndarray, components_data: List[Dict], name: str, step_dir: Path) -> List[Dict]:
    """Extract clean product images from component coordinates using pink button removal"""
    print("🖼️ Extracting clean product images from component coordinates")

    clean_products = []

    for i, component in enumerate(components_data):
        try:
            print(f"   🎯 Processing component {i+1}/{len(components_data)}")

            # Get product tile region (without text area)
            product_x = int(component["product_image_x"])
            product_y = int(component["product_image_y"])
            product_width = int(component["product_image_width"])
            product_height = int(component["product_image_height"])

            # Extract product tile
            product_tile = original_image[product_y:product_y+product_height, product_x:product_x+product_width]

            # STEP 4A: Create hole-punched image (transparent holes where buttons were)
            hole_punched_tile = _remove_pink_button_with_holes(product_tile.copy(), component)

            # Save hole-punched product image (intermediate step)
            holes_filename = f"{name}_component_{i+1}_product_holes.png"
            holes_path = step_dir / holes_filename if step_dir else holes_filename
            if step_dir:
                success = cv2.imwrite(str(holes_path), hole_punched_tile)
                if not success:
                    print(f"      ❌ Failed to save hole-punched image: {holes_filename}")
                    continue

            # Save clean product image (for compatibility)
            product_filename = f"{name}_component_{i+1}_clean_product.png"
            product_path = step_dir / product_filename if step_dir else product_filename
            if step_dir:
                cv2.imwrite(str(product_path), hole_punched_tile)

            # Extract text region (full)
            text_x = int(component["text_area_x"])
            text_y = int(component["text_area_y"])
            text_width = int(component["text_area_width"])
            text_height = int(component["text_area_height"])

            text_region = original_image[text_y:text_y+text_height, text_x:text_x+text_width]
            text_filename = f"{name}_component_{i+1}_text_region.png"
            text_path = step_dir / text_filename if step_dir else text_filename
            if step_dir:
                cv2.imwrite(str(text_path), text_region)

            # FIX: Dynamically detect text lines from bottom-up to handle variable product name lengths
            # Product 4 has 1-line name, Products 1-3 have 2-line names
            # Bottom-up detection ensures we always get the correct per-unit price line

            # Detect all text lines
            detected_lines = _detect_text_lines_bottom_up(text_region)
            print(f"      🔍 Detected {len(detected_lines)} text lines in product {i+1}")

            # Extract actual bottom line (per-unit price) - DYNAMIC
            bottom_region = _extract_bottom_line_crop(text_region, padding=5)

            # Extract actual top line (main price) - DYNAMIC
            top_region = _extract_top_line_crop(text_region, padding=5)

            # Middle region: everything between top and bottom lines
            if len(detected_lines) >= 3:
                # Use detected line positions for middle section
                top_line = detected_lines[-1]  # Top line (last in reversed list)
                bottom_line = detected_lines[0]  # Bottom line (first in reversed list)
                middle_start = top_line['y_end'] + 5
                middle_end = bottom_line['y_start'] - 5
                middle_region = text_region[middle_start:middle_end, :]
            else:
                # Fallback for products with few lines
                middle_region = text_region[top_region.shape[0]:-bottom_region.shape[0], :]

            # Save separate regions
            top_text_filename = f"{name}_component_{i+1}_text_top.png"
            middle_text_filename = f"{name}_component_{i+1}_text_middle.png"
            bottom_text_filename = f"{name}_component_{i+1}_text_bottom.png"

            top_text_path = step_dir / top_text_filename if step_dir else top_text_filename
            middle_text_path = step_dir / middle_text_filename if step_dir else middle_text_filename
            bottom_text_path = step_dir / bottom_text_filename if step_dir else bottom_text_filename

            if step_dir:
                cv2.imwrite(str(top_text_path), top_region)
                cv2.imwrite(str(middle_text_path), middle_region)
                cv2.imwrite(str(bottom_text_path), bottom_region)
                print(f"      ✅ Saved text regions: top ({top_region.shape[0]}px), middle ({middle_region.shape[0]}px), bottom ({bottom_region.shape[0]}px) - DYNAMIC")

            # Create clean product data entry
            clean_product = {
                "component_id": component["component_id"],
                "clean_image_path": str(product_path),
                "holes_image_path": str(holes_path),  # NEW: hole-punched image path
                "text_image_path": str(text_path),  # Full text region (legacy compatibility)
                "text_top_path": str(top_text_path),  # NEW: Top region for main price (Step 5b)
                "text_middle_path": str(middle_text_path),  # NEW: Middle region for product name (Step 5a)
                "text_bottom_path": str(bottom_text_path),  # NEW: Bottom region for per-unit price (Step 5c)
                "product_region": {
                    "x": product_x, "y": product_y,
                    "width": product_width, "height": product_height
                },
                "text_region": {
                    "x": text_x, "y": text_y,
                    "width": text_width, "height": text_height
                },
                "pink_button_removed": component.get("pink_button_detected", False),
                "canvas_info": {
                    "x": component["canvas_x"], "y": component["canvas_y"],
                    "width": component["canvas_width"], "height": component["canvas_height"]
                }
            }

            clean_products.append(clean_product)
            print(f"      ✅ Saved: {product_filename} and {text_filename}")

        except Exception as e:
            print(f"      ❌ Failed to extract component {i+1}: {e}")
            continue

    print(f"   🎉 Successfully extracted {len(clean_products)} clean product images")
    return clean_products


def _detect_text_lines_bottom_up(text_region: np.ndarray) -> List[Dict]:
    """
    Detect text lines from bottom to top using horizontal projection.
    Returns list of line regions with their Y coordinates.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold to isolate text
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Horizontal projection: sum pixels in each row
    h_projection = np.sum(binary, axis=1)

    # Find rows with text (non-zero projection)
    # Use a threshold to ignore noise
    threshold = np.max(h_projection) * 0.1
    text_rows = h_projection > threshold

    # Find contiguous text blocks (lines)
    lines = []
    in_line = False
    line_start = 0

    for i, has_text in enumerate(text_rows):
        if has_text and not in_line:
            # Start of new line
            line_start = i
            in_line = True
        elif not has_text and in_line:
            # End of line
            line_end = i
            lines.append({
                'y_start': line_start,
                'y_end': line_end,
                'height': line_end - line_start,
                'center_y': (line_start + line_end) // 2
            })
            in_line = False

    # Handle case where last line extends to bottom
    if in_line:
        lines.append({
            'y_start': line_start,
            'y_end': len(text_rows),
            'height': len(text_rows) - line_start,
            'center_y': (line_start + len(text_rows)) // 2
        })

    # Reverse to get bottom-first order
    lines.reverse()

    return lines

def _extract_bottom_line_crop(text_region: np.ndarray, padding: int = 10) -> np.ndarray:
    """
    Extract the bottom-most text line with padding.
    Uses bottom-up line detection to find actual last line.
    """
    lines = _detect_text_lines_bottom_up(text_region)

    if not lines:
        # Fallback: return bottom 60px
        return text_region[-60:, :]

    # Get the bottom-most line (first in reversed list)
    bottom_line = lines[0]

    # Add padding above and below
    y_start = max(0, bottom_line['y_start'] - padding)
    y_end = min(text_region.shape[0], bottom_line['y_end'] + padding)

    # Extract crop
    crop = text_region[y_start:y_end, :]

    return crop

def _extract_top_line_crop(text_region: np.ndarray, padding: int = 10) -> np.ndarray:
    """
    Extract the top-most text line with padding.
    Uses bottom-up line detection to find actual first line.
    """
    lines = _detect_text_lines_bottom_up(text_region)

    if not lines:
        # Fallback: return top 60px
        return text_region[0:60, :]

    # Get the top-most line (last in reversed list)
    top_line = lines[-1]

    # Add padding above and below
    y_start = max(0, top_line['y_start'] - padding)
    y_end = min(text_region.shape[0], top_line['y_end'] + padding)

    # Extract crop
    crop = text_region[y_start:y_end, :]

    return crop

def _remove_pink_button_with_holes(tile_image, component_data):
    """
    STEP 4A: Remove pink buttons by creating transparent circular holes using precise coordinates.
    This creates the hole-punched intermediate image as per original pipeline.
    Uses ONLY the detected button coordinates - no color filtering.
    """
    if tile_image is None or tile_image.size == 0:
        return tile_image

    h, w = tile_image.shape[:2]

    # Convert to BGRA for transparency support
    if tile_image.shape[2] == 3:
        tile_image = cv2.cvtColor(tile_image, cv2.COLOR_BGR2BGRA)

    # Use ONLY the precise coordinates from button detection - no color filtering
    if component_data.get('pink_button_detected', False):
        # Calculate button position relative to the product tile
        # Component data has button coordinates in canvas space, need to convert to tile space

        # Get canvas and product regions to calculate offset
        canvas_x = int(component_data.get('canvas_x', 0))
        canvas_y = int(component_data.get('canvas_y', 0))
        product_x = int(component_data.get('product_image_x', 0))
        product_y = int(component_data.get('product_image_y', 0))

        # Button coordinates in canvas space
        button_center_x = int(component_data.get('pink_button_center_x', 0))
        button_center_y = int(component_data.get('pink_button_center_y', 0))

        # Convert to product tile space
        tile_button_x = button_center_x - product_x
        tile_button_y = button_center_y - product_y

        # Ensure coordinates are within tile bounds
        if 0 <= tile_button_x < w and 0 <= tile_button_y < h:
            # Create a circular mask to completely cut out the button area
            radius = 60  # Increased radius to ensure complete removal including any pink rim

            # Create mask for the circular region to cut out
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (tile_button_x, tile_button_y), radius, 255, -1)

            # Cut out the circular region by setting all channels to 0 (including alpha if present)
            if tile_image.shape[2] == 4:  # BGRA
                tile_image[mask > 0] = [0, 0, 0, 0]  # Fully transparent
            else:  # BGR - convert to BGRA first
                tile_image = cv2.cvtColor(tile_image, cv2.COLOR_BGR2BGRA)
                tile_image[mask > 0] = [0, 0, 0, 0]  # Fully transparent

            print(f"      ✅ Cut out circular area at ({tile_button_x}, {tile_button_y}) with radius {radius}")
        else:
            print(f"      ⚠️ Button coordinates ({tile_button_x}, {tile_button_y}) outside tile bounds {w}x{h}")
    else:
        print("      ⚠️ No button detected for this component - no hole punching needed")

    return tile_image




def _initialize_background_removal_manager():
    """Initialize background removal manager with proper error handling"""
    try:
        import sys
        import os

        # Add src directory to Python path
        src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if src_path not in sys.path:
            sys.path.append(src_path)

        from src.background_removal_manager import BackgroundRemovalManager
        manager = BackgroundRemovalManager()
        print("   ✅ Background removal manager initialized")
        return manager
    except Exception as e:
        print(f"   ⚠️  Background removal manager unavailable: {e}")
        return None


def _apply_background_removal(input_image_path: str, bg_removal_manager, output_dir: Path) -> Dict[str, Any]:
    """Apply background removal to clean product image"""
    try:
        if not os.path.exists(input_image_path):
            return {"success": False, "error": "Input image not found"}

        # Load the clean product image
        clean_product_image = cv2.imread(input_image_path)
        if clean_product_image is None:
            return {"success": False, "error": "Failed to load input image"}

        print(f"      🔄 Removing background from {os.path.basename(input_image_path)}...")

        # Create temporary files for background removal
        temp_input = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        cv2.imwrite(temp_input.name, clean_product_image)
        temp_input.close()

        # Create temp output file
        temp_output = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_output.close()

        try:
            # Use proper background removal method
            result = bg_removal_manager.process_with_fallback(temp_input.name, temp_output.name)

            # Check if result has success attribute or if output file exists
            success = (hasattr(result, 'success') and result.success) or os.path.exists(temp_output.name)

            if success and os.path.exists(temp_output.name):
                # Load the background-removed image
                product_nobg = cv2.imread(temp_output.name, cv2.IMREAD_UNCHANGED)
                if product_nobg is None:
                    raise Exception("Failed to load background-removed image")

                # Generate output filename
                base_name = os.path.basename(input_image_path).replace('_clean_product.png', '')
                nobg_path = output_dir / f"{base_name}_product_nobg.png"  # PNG for transparency
                cv2.imwrite(str(nobg_path), product_nobg)

                print(f"      ✅ Background removed successfully")
                print(f"      📁 Saved: {nobg_path}")

                return {
                    "success": True,
                    "output_path": str(nobg_path),
                    "temp_input": temp_input.name,
                    "temp_output": temp_output.name
                }
            else:
                error_msg = "Unknown error"
                if hasattr(result, 'error_message'):
                    error_msg = result.error_message
                elif hasattr(result, 'message'):
                    error_msg = result.message
                elif hasattr(result, '__dict__'):
                    error_msg = str(result.__dict__)
                raise Exception(f"Background removal failed: {error_msg}")

        finally:
            # Clean up temp files
            try:
                os.unlink(temp_input.name)
                os.unlink(temp_output.name)
            except:
                pass

    except Exception as e:
        print(f"      ❌ Background removal failed: {e}")
        return {"success": False, "error": str(e)}