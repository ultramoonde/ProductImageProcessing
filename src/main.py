#!/usr/bin/env python3
"""
Food Delivery Product Image Extraction Tool

Extracts product tiles and metadata from food delivery app screenshots.
Handles Flink and Dr. Morris app layouts with 4 products per viewport.
"""

import argparse
import os
import cv2
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import sys

from tile_detector import TileDetector
from image_processor import ImageProcessor
from text_extractor import TextExtractor
# Import AI vision analysis system
try:
    from real_vision_analyzer import RealVisionAnalyzer
    from ai_text_analyzer import AITextAnalyzer
    AI_ANALYSIS_AVAILABLE = True
except ImportError:
    AI_ANALYSIS_AVAILABLE = False
    print("⚠️ AI analysis system not available - using basic OCR only")

class FoodExtractor:
    def __init__(self, output_dir: str, debug: bool = False, 
                 enable_background_removal: bool = True,
                 bg_config_path: Optional[str] = None,
                 bg_strategy: str = 'cost_first',
                 bg_provider: Optional[str] = None,
                 enable_ai_analysis: bool = True):
        """Initialize the food extraction system."""
        self.output_dir = Path(output_dir)
        self.debug = debug
        
        # Create output directories
        self.images_dir = self.output_dir / "extracted_images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components with background removal settings
        self.tile_detector = TileDetector()
        self.image_processor = ImageProcessor(
            enable_background_removal=enable_background_removal,
            bg_config_path=bg_config_path
        )
        self.text_extractor = TextExtractor()
        
        # Initialize AI analysis system with cost-optimized consensus
        self.enable_ai_analysis = enable_ai_analysis and AI_ANALYSIS_AVAILABLE
        self.use_consensus = True  # Enable cost-optimized consensus voting
        
        if self.enable_ai_analysis:
            try:
                if self.use_consensus:
                    from local_consensus_analyzer import LocalConsensusAnalyzer
                    
                    # Initialize cost-optimized consensus system
                    self.ai_analyzer = LocalConsensusAnalyzer(
                        use_api_fallback=True,
                        fallback_provider="openai"  # GPT-4o-mini is 20x cheaper than Claude
                    )
                    self.vision_analyzer = None  # Not needed with consensus system
                    print("🏆 Cost-optimized consensus system initialized (local + API fallback)")
                else:
                    self.vision_analyzer = RealVisionAnalyzer()
                    # Force regex parsing for now since Ollama is not running
                    self.ai_analyzer = AITextAnalyzer(use_openai=False)  
                    print("✅ AI vision analysis system initialized (with regex fallback)")
            except Exception as e:
                print(f"⚠️ AI analysis initialization failed: {e}")
                self.enable_ai_analysis = False
                self.vision_analyzer = None
                self.ai_analyzer = None
        else:
            self.vision_analyzer = None
            self.ai_analyzer = None
        
        # Setup logging first
        self._setup_logging()
        
        # Configure background removal
        if enable_background_removal:
            if bg_strategy:
                self.image_processor.set_background_removal_strategy(bg_strategy)
            
            # Log background removal settings
            if self.image_processor.bg_manager:
                cost_estimate = self.image_processor.estimate_background_removal_cost(100)
                if cost_estimate:
                    self.logger.info(f"Background removal configured: {bg_strategy}")
                    self.logger.info(f"Primary provider: {cost_estimate.get('primary_provider', 'unknown')}")
                    self.logger.info(f"Estimated cost per 100 images: ${cost_estimate.get('total_cost_estimate', 0):.2f}")
                else:
                    self.logger.info(f"Background removal enabled with strategy: {bg_strategy}")
        
        # Statistics
        self.stats = {
            'screenshots_processed': 0,
            'tiles_detected': 0,
            'tiles_extracted': 0,
            'ocr_success': 0,
            'errors': 0
        }
        
        # Results storage
        self.all_products = []
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / "processing_log.txt"
        
        # Create logger
        self.logger = logging.getLogger('FoodExtractor')
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def process_screenshot(self, screenshot_path: str) -> bool:
        """
        Process a single screenshot and extract all product tiles.
        Returns True if processing was successful.
        """
        try:
            self.logger.info(f"Processing screenshot: {screenshot_path}")
            
            # Load image
            image = cv2.imread(screenshot_path)
            if image is None:
                self.logger.error(f"Could not load image: {screenshot_path}")
                self.stats['errors'] += 1
                return False
            
            screenshot_name = Path(screenshot_path).stem
            
            # Extract category from header
            header_region = self.image_processor.extract_header_region(image)
            category, subcategory = self.text_extractor.extract_category_from_header(header_region)
            
            self.logger.info(f"Detected category: {category}, subcategory: {subcategory}")
            
            # Detect tiles
            tiles = self.tile_detector.detect_tiles(image)
            
            if not tiles:
                # Try alternative detection method
                self.logger.warning(f"No tiles detected with primary method, trying template matching...")
                tiles = self.tile_detector.detect_tiles_template_matching(image)
            
            self.stats['tiles_detected'] += len(tiles)
            self.logger.info(f"Detected {len(tiles)} tiles in {screenshot_name}")
            
            if self.debug and tiles:
                # Save debug visualization
                debug_image = self.tile_detector.visualize_detections(image, tiles)
                debug_path = self.output_dir / f"{screenshot_name}_debug.png"
                cv2.imwrite(str(debug_path), debug_image)
                self.logger.debug(f"Saved debug visualization: {debug_path}")
            
            # Process each detected tile
            for i, (x, y, w, h) in enumerate(tiles):
                self._process_single_tile(
                    image, x, y, w, h, 
                    screenshot_name, category, subcategory, i
                )
            
            self.stats['screenshots_processed'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing screenshot {screenshot_path}: {str(e)}")
            self.stats['errors'] += 1
            return False
    
    def _process_single_tile(self, image, x: int, y: int, w: int, h: int, 
                           screenshot_name: str, category: str, subcategory: str, tile_index: int):
        """Process a single detected tile."""
        try:
            # STEP 1: Extract raw tile from screenshot (full size)
            raw_tile = image[y:y+h, x:x+w]
            self.logger.debug(f"🎯 Extracted raw tile {tile_index+1} at full size: {raw_tile.shape}")
            
            # STEP 2: Remove pink + button from raw tile BEFORE any processing
            self.logger.debug(f"🎯 Attempting to remove pink button from tile {tile_index+1}")
            raw_tile = self._remove_pink_button_from_tile(raw_tile)
            self.logger.debug(f"🎯 Button removal complete for tile {tile_index+1}")
            
            # STEP 3: Extract text region for AI analysis (660x240px below tile)
            text_region = self._extract_text_region(image, x, y, w, h)
            
            # STEP 4: AI analysis if available
            if self.enable_ai_analysis and text_region is not None:
                self.logger.debug(f"🤖 Performing AI analysis on tile {tile_index+1}")
                ai_analysis = self._perform_ai_analysis(raw_tile, text_region)
            else:
                ai_analysis = None
                
            # STEP 5: Now process the tile (resize, rounded corners, etc.)
            tile = self.image_processor.process_extracted_tile(raw_tile)
            
            # Generate filename for tile
            tile_filename = f"{screenshot_name}_tile_{tile_index+1}.png"
            tile_path = self.images_dir / tile_filename
            
            # STEP 6: Save tile image with background removal (button already removed)
            if self.image_processor.save_tile_as_png(tile, str(tile_path)):
                self.stats['tiles_extracted'] += 1
                self.logger.debug(f"Saved tile: {tile_filename}")
                
                # Extract text data - use AI analysis if available, fallback to OCR
                if ai_analysis:
                    text_data = ai_analysis  # Rich AI-extracted data
                    self.logger.debug(f"🤖 Using AI-extracted data for tile {tile_index+1}")
                else:
                    text_data = self.text_extractor.extract_text_from_tile(tile)  # Fallback OCR
                    self.logger.debug(f"📝 Using OCR-extracted data for tile {tile_index+1}")
                
                # Validate data (AI data is generally more reliable)
                if ai_analysis or self.text_extractor.validate_extracted_data(text_data):
                    self.stats['ocr_success'] += 1
                    
                    # Clean up product name for filename if we have one
                    product_name = text_data.get('product_name', f'product_{tile_index+1}')
                    clean_name = self._clean_filename(product_name)
                    
                    # Generate final filename
                    final_filename = f"{category}_{subcategory}_{clean_name}_{tile_index+1}.png"
                    final_filename = self._clean_filename(final_filename)
                    final_path = self.images_dir / final_filename
                    
                    # Rename file if different
                    if final_filename != tile_filename:
                        try:
                            tile_path.rename(final_path)
                            tile_filename = final_filename
                            self.logger.debug(f"Renamed tile to: {final_filename}")
                        except:
                            # Keep original name if rename fails
                            final_filename = tile_filename
                    
                    # Store comprehensive product data (AI-enhanced or OCR)
                    product_data = {
                        # Basic identification
                        'image_filename': final_filename,
                        'product_name': text_data.get('product_name', ''),
                        'brand': text_data.get('brand', ''),
                        'manufacturer': text_data.get('manufacturer', ''),
                        
                        # Pricing information
                        'price': text_data.get('price', ''),
                        'price_per_unit': text_data.get('price_per_unit', ''),
                        'cost_per_unit': text_data.get('cost_per_unit', ''),
                        'discount': text_data.get('discount', ''),
                        
                        # Product details
                        'weight': text_data.get('weight', ''),
                        'quantity': text_data.get('quantity', ''),
                        'unit': text_data.get('unit', ''),
                        'additional_info': text_data.get('additional_info', ''),
                        
                        # Categories
                        'category': category,
                        'subcategory': subcategory,
                        
                        # AI analysis metadata
                        'ai_confidence': text_data.get('ai_confidence', ''),
                        'analysis_method': 'AI_Vision' if ai_analysis else 'OCR',
                        
                        # Technical metadata
                        'source_screenshot': screenshot_name,
                        'extraction_timestamp': datetime.now().isoformat(),
                        'tile_position': f"x={x}, y={y}, w={w}, h={h}"
                    }
                    
                    self.all_products.append(product_data)
                    
                    self.logger.info(f"Successfully extracted: {text_data.get('product_name', 'Unknown')} - {text_data.get('price', 'No price')}")
                    
                    # Debug OCR if enabled
                    if self.debug:
                        debug_ocr_path = self.output_dir / f"{screenshot_name}_tile_{tile_index+1}_ocr_debug.png"
                        self.text_extractor.debug_ocr_results(tile, str(debug_ocr_path))
                else:
                    self.logger.warning(f"OCR validation failed for tile {tile_index+1} in {screenshot_name}")
                    # Still save the product data with empty fields (for failed OCR/AI)
                    product_data = {
                        # Basic identification
                        'image_filename': tile_filename,
                        'product_name': text_data.get('product_name', '') if text_data else '',
                        'brand': text_data.get('brand', '') if text_data else '',
                        'manufacturer': text_data.get('manufacturer', '') if text_data else '',
                        
                        # Pricing information
                        'price': text_data.get('price', '') if text_data else '',
                        'price_per_unit': text_data.get('price_per_unit', '') if text_data else '',
                        'cost_per_unit': text_data.get('cost_per_unit', '') if text_data else '',
                        'discount': text_data.get('discount', '') if text_data else '',
                        
                        # Product details
                        'weight': text_data.get('weight', '') if text_data else '',
                        'quantity': text_data.get('quantity', '') if text_data else '',
                        'unit': text_data.get('unit', '') if text_data else '',
                        'additional_info': text_data.get('additional_info', '') if text_data else '',
                        
                        # Categories
                        'category': category,
                        'subcategory': subcategory,
                        
                        # AI analysis metadata
                        'ai_confidence': '',
                        'analysis_method': 'FAILED',
                        
                        # Technical metadata
                        'source_screenshot': screenshot_name,
                        'extraction_timestamp': datetime.now().isoformat(),
                        'tile_position': f"x={x}, y={y}, w={w}, h={h}"
                    }
                    self.all_products.append(product_data)
            else:
                self.logger.error(f"Failed to save tile {tile_index+1} from {screenshot_name}")
                self.stats['errors'] += 1
                
        except Exception as e:
            self.logger.error(f"Error processing tile {tile_index+1} from {screenshot_name}: {str(e)}")
            self.stats['errors'] += 1
    
    def _remove_pink_button_from_tile(self, tile_image):
        """
        Remove pink + button from raw tile image before background removal.
        This is the correct workflow: extract tile -> remove button -> background removal.
        """
        import cv2
        import numpy as np
        
        if tile_image is None or tile_image.size == 0:
            self.logger.debug("⚠️ Tile image is None or empty, skipping button removal")
            return tile_image
        
        h, w = tile_image.shape[:2]
        self.logger.debug(f"🎯 Processing tile of size {w}x{h} for button removal")
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(tile_image, cv2.COLOR_BGR2HSV)
        
        # Create removal mask
        removal_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Method 1: Precise pink color targeting for Flink button
        # Actual button color found: HSV=[167, 228, 226] HEX=#E2186F
        # Create range around this specific color
        lower_pink1 = np.array([160, 200, 200])  # H=160-175, high saturation, high value
        upper_pink1 = np.array([175, 255, 255])
        
        # Also check for slightly different pink variations
        lower_pink2 = np.array([165, 180, 180])  # Slightly broader range
        upper_pink2 = np.array([170, 255, 255])
        
        pink_mask1 = cv2.inRange(hsv, lower_pink1, upper_pink1)
        pink_mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
        pink_mask = cv2.bitwise_or(pink_mask1, pink_mask2)
        
        # Method 2: Circular button detection
        gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
            param1=50, param2=35, minRadius=35, maxRadius=65
        )
        
        button_found = False
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Button typically in bottom-right area
                if x > w * 0.75 and y > h * 0.75:
                    cv2.circle(removal_mask, (x, y), r + 8, 255, -1)  # Small padding
                    self.logger.debug(f"🎯 Removed + button from tile at ({x}, {y}) radius {r}")
                    button_found = True
        
        # Method 3: Fallback - target expected button position for 573x573 tiles
        if not button_found and 400 < w < 700 and 400 < h < 700:
            button_x = w - 48
            button_y = h - 48
            button_radius = 28
            
            if 0 < button_x < w and 0 < button_y < h:
                # Check if pink content exists at expected position
                center_color = hsv[button_y, button_x]
                if (160 <= center_color[0] <= 175) and center_color[1] > 180 and center_color[2] > 180:
                    cv2.circle(removal_mask, (button_x, button_y), button_radius, 255, -1)
                    self.logger.debug(f"🎯 Removed fallback + button at ({button_x}, {button_y})")
                    button_found = True
        
        # Restrict pink removal to bottom-right area only
        pink_mask_restricted = np.zeros_like(pink_mask)
        pink_mask_restricted[int(h*0.7):h, int(w*0.7):w] = pink_mask[int(h*0.7):h, int(w*0.7):w]
        
        # Combine masks
        combined_mask = cv2.bitwise_or(removal_mask, pink_mask_restricted)
        
        # Apply removal by setting pixels to neutral color (will be removed by background removal later)
        if np.sum(combined_mask > 0) > 0:
            # Set button area to white (neutral background color)
            tile_image[combined_mask > 0] = [245, 245, 245]  # Light gray/white
            
            removed_pixels = np.sum(combined_mask > 0)
            removal_percentage = (removed_pixels / (h * w)) * 100
            if removal_percentage > 0.1:
                self.logger.debug(f"🎯 Removed {removal_percentage:.1f}% pink button pixels from tile")
        
        return tile_image
    
    def _extract_text_region(self, image, tile_x: int, tile_y: int, tile_w: int, tile_h: int):
        """
        Extract text region below the tile for AI analysis.
        Returns 660x240px region containing product text information.
        """
        try:
            img_height, img_width = image.shape[:2]
            
            # Calculate text region position (below the tile)
            text_region_x = max(0, tile_x - 40)  # Start slightly left of tile
            text_region_y = tile_y + tile_h + 10  # Start below tile
            text_region_w = 660  # Standard text region width
            text_region_h = 240  # Standard text region height
            
            # Ensure we don't go beyond image boundaries
            text_region_x = min(text_region_x, img_width - text_region_w)
            text_region_y = min(text_region_y, img_height - text_region_h)
            
            if text_region_x < 0 or text_region_y < 0:
                self.logger.debug("Text region would be outside image bounds")
                return None
                
            # Extract text region
            text_region = image[text_region_y:text_region_y+text_region_h, 
                             text_region_x:text_region_x+text_region_w]
            
            self.logger.debug(f"Extracted text region: {text_region.shape} at ({text_region_x}, {text_region_y})")
            return text_region
            
        except Exception as e:
            self.logger.error(f"Error extracting text region: {e}")
            return None
    
    def _perform_ai_analysis(self, tile_image, text_region):
        """
        Perform AI analysis on tile and text region to extract rich product data.
        Uses consensus voting system or fallback to single model analysis.
        """
        try:
            if not self.ai_analyzer:
                return None
            
            if self.use_consensus:
                # Use consensus system (async method)
                import asyncio
                
                # Create async wrapper for consensus analysis
                async def run_consensus():
                    return await self.ai_analyzer.analyze_product_with_consensus(tile_image, text_region)
                
                # Run consensus analysis
                ai_result = asyncio.run(run_consensus())
                
                if ai_result and any(ai_result.values()):
                    analysis_method = ai_result.get('analysis_method', 'unknown')
                    consensus_confidence = ai_result.get('consensus_confidence', 0.0)
                    
                    self.logger.debug(f"🏆 Consensus analysis successful: {ai_result.get('product_name', 'Unknown')} ({analysis_method})")
                    if 'local_consensus' in analysis_method:
                        self.logger.debug(f"💰 Cost saved using local consensus ({consensus_confidence:.1%} confidence)")
                    elif 'api_fallback' in analysis_method:
                        self.logger.debug(f"🌐 Used API fallback after local consensus failed")
                    
                    return ai_result
                else:
                    self.logger.debug("🏆 Consensus analysis failed")
                    return None
                    
            else:
                # Use single model approach (legacy)
                ai_result = self.ai_analyzer.analyze_product_tile_and_text(tile_image, text_region)
                
                if ai_result and any(ai_result.values()):
                    self.logger.debug(f"🤖 AI vision analysis successful: {ai_result.get('product_name', 'Unknown')}")
                    return ai_result
                else:
                    # Fallback: Use sophisticated OCR parsing when AI vision fails
                    self.logger.debug("🤖 AI vision failed, using intelligent OCR parsing...")
                    ocr_result = self.ai_analyzer.analyze_product_text_region(text_region)
                    
                    if ocr_result and any(ocr_result.values()):
                        self.logger.debug(f"📝 OCR parsing successful: {ocr_result.get('product_name', 'Unknown')}")
                        return ocr_result
                    else:
                        self.logger.debug("📝 OCR parsing also failed")
                        return None
                
        except Exception as e:
            self.logger.error(f"AI analysis error: {e}")
            # Last resort: try basic OCR parsing
            try:
                if text_region is not None:
                    ocr_fallback = self.ai_analyzer.analyze_product_text_region(text_region)
                    if ocr_fallback:
                        self.logger.debug("📝 Emergency OCR fallback successful")
                        return ocr_fallback
            except:
                pass
            return None
    
    def _clean_filename(self, filename: str) -> str:
        """Clean filename to be filesystem safe."""
        import re
        # Replace spaces with underscores and remove special characters
        cleaned = re.sub(r'[^\w\-_.]', '_', filename)
        # Remove multiple underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        # Limit length
        if len(cleaned) > 50:
            cleaned = cleaned[:50]
        return cleaned or 'unnamed'
    
    def save_metadata_csv(self):
        """Save all extracted product data to CSV."""
        if not self.all_products:
            self.logger.warning("No products to save")
            return
        
        csv_path = self.output_dir / "products_metadata.csv"
        df = pd.DataFrame(self.all_products)
        
        # Reorder columns for comprehensive data display
        column_order = [
            # Basic identification
            'image_filename', 'product_name', 'brand', 'manufacturer',
            
            # Pricing information  
            'price', 'price_per_unit', 'cost_per_unit', 'discount',
            
            # Product details
            'weight', 'quantity', 'unit', 'additional_info',
            
            # Categories
            'category', 'subcategory',
            
            # AI metadata
            'ai_confidence', 'analysis_method',
            
            # Technical metadata
            'source_screenshot', 'extraction_timestamp', 'tile_position'
        ]
        
        df = df.reindex(columns=column_order)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        self.logger.info(f"Saved metadata for {len(self.all_products)} products to {csv_path}")
    
    def print_statistics(self):
        """Print processing statistics."""
        print("\n" + "="*60)
        print("EXTRACTION STATISTICS")
        print("="*60)
        print(f"Screenshots processed: {self.stats['screenshots_processed']}")
        print(f"Tiles detected: {self.stats['tiles_detected']}")
        print(f"Tiles extracted: {self.stats['tiles_extracted']}")
        print(f"Successful OCR extractions: {self.stats['ocr_success']}")
        print(f"Errors encountered: {self.stats['errors']}")
        
        if self.stats['tiles_detected'] > 0:
            detection_rate = (self.stats['tiles_extracted'] / self.stats['tiles_detected']) * 100
            print(f"Tile extraction success rate: {detection_rate:.1f}%")
        
        if self.stats['tiles_extracted'] > 0:
            ocr_rate = (self.stats['ocr_success'] / self.stats['tiles_extracted']) * 100
            print(f"OCR success rate: {ocr_rate:.1f}%")
        
        print("="*60)
    
    def process_batch(self, input_dir: str) -> bool:
        """Process all screenshots in the input directory."""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            self.logger.error(f"Input directory does not exist: {input_dir}")
            return False
        
        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            self.logger.error(f"No image files found in {input_dir}")
            return False
        
        self.logger.info(f"Found {len(image_files)} screenshots to process")
        
        # Process each screenshot
        success_count = 0
        for image_file in image_files:
            if self.process_screenshot(str(image_file)):
                success_count += 1
        
        # Save results
        self.save_metadata_csv()
        
        self.logger.info(f"Batch processing complete: {success_count}/{len(image_files)} screenshots processed successfully")
        
        # Display consensus statistics if using consensus system
        if self.use_consensus and self.ai_analyzer and hasattr(self.ai_analyzer, 'print_stats_summary'):
            self.ai_analyzer.print_stats_summary()
        
        return success_count > 0

def main():
    parser = argparse.ArgumentParser(
        description="Extract product images and data from food delivery app screenshots"
    )
    
    parser.add_argument(
        '--input', '-i', 
        required=True,
        help="Input directory containing screenshots"
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True, 
        help="Output directory for extracted images and metadata"
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug mode with additional output"
    )
    
    parser.add_argument(
        '--bg-removal',
        choices=['auto', 'rembg', 'remove_bg', 'photoroom', 'removal_ai', 'off'],
        default='auto',
        help="Background removal strategy (default: auto)"
    )
    
    parser.add_argument(
        '--bg-strategy', 
        choices=['cost_first', 'quality_first', 'speed_first', 'balanced', 'production'],
        default='cost_first',
        help="Background removal strategy (default: cost_first)"
    )
    
    parser.add_argument(
        '--bg-config',
        type=str,
        help="Path to background removal configuration file"
    )
    
    parser.add_argument(
        '--ai-analysis',
        action='store_true', 
        default=True,
        help="Enable AI vision analysis for rich text extraction (default: True)"
    )
    
    parser.add_argument(
        '--no-ai-analysis',
        dest='ai_analysis',
        action='store_false',
        help="Disable AI analysis and use basic OCR only"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Configure background removal
    enable_bg_removal = args.bg_removal != 'off'
    
    # Initialize and run extractor with background removal and AI analysis settings
    extractor = FoodExtractor(
        args.output, 
        debug=args.debug,
        enable_background_removal=enable_bg_removal,
        bg_config_path=args.bg_config,
        bg_strategy=args.bg_strategy,
        bg_provider=args.bg_removal if args.bg_removal != 'auto' else None,
        enable_ai_analysis=args.ai_analysis
    )
    
    print(f"Starting food delivery product extraction...")
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    print(f"Background removal: {args.bg_removal} (strategy: {args.bg_strategy})")
    print(f"AI analysis: {'ON' if args.ai_analysis else 'OFF'}")
    print("-" * 60)
    
    success = extractor.process_batch(args.input)
    
    # Print final statistics
    extractor.print_statistics()
    
    # Print background removal statistics if available
    if extractor.image_processor.bg_manager:
        bg_stats = extractor.image_processor.get_background_removal_stats()
        if bg_stats and bg_stats['total_processed'] > 0:
            print(f"\n🎨 Background Removal Statistics:")
            print(f"Total processed: {bg_stats['total_processed']}")
            print(f"Success rate: {bg_stats['successful']}/{bg_stats['total_processed']} ({bg_stats['successful']/bg_stats['total_processed']*100:.1f}%)")
            print(f"Total cost: ${bg_stats['total_cost']:.3f}")
            print(f"Average quality: {bg_stats['average_quality']:.3f}")
            
            if bg_stats['provider_usage']:
                print(f"Provider usage:")
                for provider, stats in bg_stats['provider_usage'].items():
                    print(f"  - {provider}: {stats['count']} images (avg quality: {stats['avg_quality']:.3f}, cost: ${stats['cost']:.3f})")
    
    if success:
        print(f"\n✅ Extraction completed successfully!")
        print(f"📁 Images saved to: {extractor.images_dir}")
        print(f"📊 Metadata saved to: {extractor.output_dir / 'products_metadata.csv'}")
        print(f"📝 Processing log: {extractor.output_dir / 'processing_log.txt'}")
        return 0
    else:
        print("\n❌ Extraction failed or no products were extracted")
        return 1

if __name__ == "__main__":
    exit(main())