#!/usr/bin/env python3
"""
Registry-Integrated Food Delivery Product Extraction Tool

Large-scale processing with file registry, status tracking, and conflict prevention.
Prevents duplicate processing and enables resumable batch jobs.
"""

import argparse
import os
import cv2
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging

# Import existing components
from tile_detector import TileDetector
from image_processor import ImageProcessor
from text_extractor import TextExtractor

# Import new registry and CSV systems
from processing_registry import ProcessingRegistry, ProcessingStatus
from enhanced_csv_exporter import EnhancedCSVExporter

# Import AI analysis if available
try:
    from local_consensus_analyzer import LocalConsensusAnalyzer
    AI_ANALYSIS_AVAILABLE = True
except ImportError:
    AI_ANALYSIS_AVAILABLE = False
    print("⚠️ AI consensus system not available - using basic OCR only")

class RegistryIntegratedExtractor:
    """Food extractor with integrated file processing registry"""
    
    def __init__(self, output_dir: str, registry_path: str = None, 
                 debug: bool = False, enable_ai_analysis: bool = True,
                 bg_strategy: str = 'cost_first'):
        """Initialize registry-integrated extraction system"""
        
        self.output_dir = Path(output_dir)
        self.debug = debug
        
        # Create output directories  
        self.images_dir = self.output_dir / "extracted_images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processing registry
        if not registry_path:
            registry_path = self.output_dir / "processing_registry.db"
        self.registry = ProcessingRegistry(str(registry_path))
        
        # Initialize enhanced CSV exporter
        csv_path = self.output_dir / "products_comprehensive.csv"
        self.csv_exporter = EnhancedCSVExporter(str(csv_path))
        
        # Initialize core components
        self.tile_detector = TileDetector()
        self.image_processor = ImageProcessor(
            enable_background_removal=True
        )
        self.text_extractor = TextExtractor()
        
        # Initialize AI analysis system
        self.enable_ai_analysis = enable_ai_analysis and AI_ANALYSIS_AVAILABLE
        if self.enable_ai_analysis:
            self.ai_analyzer = LocalConsensusAnalyzer(
                use_api_fallback=False  # Keep costs at $0
            )
            print("🏆 AI consensus system initialized")
        else:
            self.ai_analyzer = None
            print("📝 Using basic OCR analysis")
            
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup comprehensive logging system"""
        log_path = self.output_dir / "processing_log.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def register_input_files(self, input_path: str) -> int:
        """Register all input files for processing"""
        input_path = Path(input_path)
        file_paths = []
        
        if input_path.is_file():
            file_paths = [str(input_path)]
        else:
            # Find all image files
            extensions = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
            for ext in extensions:
                file_paths.extend([str(p) for p in input_path.glob(ext)])
        
        registered_count = self.registry.register_files(file_paths)
        self.logger.info(f"Registered {registered_count} files for processing")
        
        return registered_count
    
    def process_single_file(self, file_info: Dict[str, Any]) -> bool:
        """Process a single file with comprehensive data extraction"""
        file_path = file_info['file_path']
        file_id = file_info['id']
        
        try:
            start_time = time.time()
            self.logger.info(f"Processing: {file_path}")
            
            # Load and validate image
            image = cv2.imread(file_path)
            if image is None:
                raise Exception(f"Could not load image: {file_path}")
            
            # Extract category information from filename/path
            category_info = self._extract_category_info(file_path)
            
            # Detect tiles
            tiles = self.tile_detector.detect_tiles(image)
            self.logger.info(f"Detected {len(tiles)} tiles in {Path(file_path).name}")
            
            if not tiles:
                raise Exception("No tiles detected in image")
            
            products_extracted = 0
            
            # Process each detected tile
            for i, (x, y, w, h) in enumerate(tiles):
                try:
                    tile_result = self._process_tile(
                        image, (x, y, w, h), i + 1, 
                        Path(file_path).name, category_info
                    )
                    
                    if tile_result:
                        self.csv_exporter.add_product(tile_result)
                        products_extracted += 1
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process tile {i+1}: {e}")
            
            # Mark as completed in registry
            processing_time = time.time() - start_time
            results = {
                'tiles_detected': len(tiles),
                'products_extracted': products_extracted,
                'processing_time_seconds': processing_time,
                'metadata': {
                    'category': category_info,
                    'image_size': f"{image.shape[1]}x{image.shape[0]}",
                    'ai_analysis_enabled': self.enable_ai_analysis
                }
            }
            
            self.registry.mark_completed(file_id, results)
            self.logger.info(f"✅ Completed {file_path}: {products_extracted} products extracted")
            return True
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            self.registry.mark_failed(file_id, error_msg)
            self.logger.error(f"❌ Failed {file_path}: {error_msg}")
            return False
    
    def _process_tile(self, image: any, tile_coords: Tuple[int, int, int, int], 
                     tile_num: int, source_file: str, category_info: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Process individual tile with comprehensive data extraction"""
        
        x, y, w, h = tile_coords
        
        # Extract tile image
        tile_image = image[y:y+h, x:x+w].copy()
        
        # Process the tile (background removal, button removal, etc.)
        processed_tile, text_region = self.image_processor.process_tile(tile_image)
        
        if processed_tile is None:
            return None
        
        # Save processed image
        image_filename = f"{category_info.get('category', 'unknown')}_{category_info.get('subcategory', 'unknown')}_{source_file.replace('.PNG', '').replace('.png', '')}_{tile_num}.png"
        image_path = self.images_dir / image_filename
        cv2.imwrite(str(image_path), processed_tile)
        
        # Extract text using OCR
        ocr_text = self.text_extractor.extract_text_from_tile(text_region if text_region is not None else processed_tile)
        
        # Initialize product data
        product_data = {
            'source_screenshot': source_file,
            'extracted_image_filename': image_filename,
            'tile_position': f"x={x}, y={y}, w={w}, h={h}",
            'category': category_info.get('category', ''),
            'subcategory': category_info.get('subcategory', ''),
            'ocr_text_raw': ocr_text,
            'extraction_method': 'AI_Consensus' if self.enable_ai_analysis else 'OCR_Basic',
            'background_removed': True,
            'button_removed': True,
            'image_width': processed_tile.shape[1],
            'image_height': processed_tile.shape[0],
            'file_size_bytes': os.path.getsize(image_path) if image_path.exists() else 0,
            'worker_id': self.registry.worker_id
        }
        
        # Enhanced AI analysis if available
        if self.enable_ai_analysis and self.ai_analyzer:
            try:
                ai_result = self.ai_analyzer.analyze_product_tile(processed_tile, text_region, ocr_text)
                
                # Merge AI results into product data
                product_data.update({
                    'product_name': ai_result.get('product_name', ''),
                    'brand': ai_result.get('brand', ''),
                    'price': ai_result.get('price', ''),
                    'weight': ai_result.get('weight', ''),
                    'quantity': ai_result.get('quantity', ''),
                    'ai_confidence_score': ai_result.get('confidence', 0),
                    'consensus_votes': ai_result.get('consensus_info', ''),
                    'nutritional_info': ai_result.get('nutrition', ''),
                    'ingredients_list': ai_result.get('ingredients', ''),
                    'product_description': ai_result.get('description', '')
                })
                
            except Exception as e:
                self.logger.warning(f"AI analysis failed for tile {tile_num}: {e}")
                product_data['extraction_errors'] = str(e)
        
        # Basic OCR fallback parsing if AI analysis unavailable
        else:
            parsed_data = self._parse_ocr_text(ocr_text)
            product_data.update(parsed_data)
        
        return product_data
    
    def _extract_category_info(self, file_path: str) -> Dict[str, str]:
        """Extract category information from filename/path"""
        filename = Path(file_path).name
        
        # Try to extract timestamp and category patterns
        import re
        
        # Pattern for timestamps like 13.34, 13:26, etc.
        time_pattern = r'(\d{1,2}[:.]\d{2})'
        category_pattern = r'(\d{2,3})'
        
        category = ""
        subcategory = ""
        
        time_match = re.search(time_pattern, filename)
        if time_match:
            category = time_match.group(1)
        
        # Extract numeric categories
        numbers = re.findall(r'\d+', filename)
        if len(numbers) >= 2:
            subcategory = numbers[-1] if len(numbers) > 1 else ""
        
        return {
            'category': category,
            'subcategory': subcategory
        }
    
    def _parse_ocr_text(self, ocr_text: str) -> Dict[str, str]:
        """Parse OCR text for basic product information"""
        import re
        
        result = {}
        
        if not ocr_text:
            return result
        
        lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
        
        # Try to extract price
        for line in lines:
            price_match = re.search(r'(\d+[,.]?\d*)\s*€', line)
            if price_match and not result.get('price'):
                result['price'] = f"{price_match.group(1)}€"
                break
        
        # Try to extract product name (usually first or second line)
        if lines:
            # Skip price-only lines
            for line in lines:
                if not re.search(r'^\d+[,.]?\d*\s*€', line) and len(line) > 3:
                    result['product_name'] = line
                    break
        
        # Try to extract weight/quantity
        for line in lines:
            weight_match = re.search(r'(\d+[,.]?\d*)\s*(g|kg|ml|l|stk|stück)', line.lower())
            if weight_match:
                result['weight'] = f"{weight_match.group(1)}{weight_match.group(2)}"
                break
        
        return result
    
    def run_batch_processing(self):
        """Run batch processing using registry system"""
        
        print("🚀 Starting registry-based batch processing...")
        
        # Print initial stats
        stats = self.registry.get_processing_stats()
        print(f"📊 Processing Status: {stats}")
        
        processed_files = 0
        failed_files = 0
        
        # Process files until queue is empty
        while True:
            # Claim next file
            file_info = self.registry.claim_next_file()
            
            if not file_info:
                print("✅ No more files to process")
                break
            
            # Process the file
            success = self.process_single_file(file_info)
            
            if success:
                processed_files += 1
            else:
                failed_files += 1
            
            # Print progress every 5 files
            if (processed_files + failed_files) % 5 == 0:
                progress = self.registry.get_processing_progress()
                print(f"📈 Progress: {progress['progress_percent']:.1f}% complete ({processed_files} success, {failed_files} failed)")
        
        # Final export and summary
        self._finalize_processing(processed_files, failed_files)
    
    def _finalize_processing(self, processed_files: int, failed_files: int):
        """Finalize processing with exports and summary"""
        
        print("📊 Finalizing processing...")
        
        # Export CSV data
        products_exported = self.csv_exporter.export_to_csv()
        
        # Export processing stats
        stats_path = self.output_dir / "processing_stats.json"
        stats = self.csv_exporter.export_summary_stats(str(stats_path))
        
        # Export registry data
        registry_csv = self.output_dir / "processing_registry.csv"
        self.registry.export_results_to_csv(str(registry_csv))
        
        # Print final summary
        print("\n🎉 PROCESSING COMPLETE!")
        print(f"📁 Files processed: {processed_files}")
        print(f"❌ Files failed: {failed_files}")
        print(f"📦 Products extracted: {products_exported}")
        print(f"📊 Average data quality: {stats['avg_data_quality_score']:.1f}%")
        print(f"⚠️  Products needing review: {stats['products_needing_review']}")
        print(f"💰 Total processing cost: $0.00")
        print(f"📂 Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Registry-Integrated Food Product Extractor')
    parser.add_argument('-i', '--input', help='Input directory or file path')
    parser.add_argument('-o', '--output', required=True, help='Output directory path')
    parser.add_argument('--registry-db', help='Custom registry database path')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-ai', action='store_true', help='Disable AI analysis')
    parser.add_argument('--bg-strategy', default='cost_first', 
                       choices=['cost_first', 'quality_first', 'speed_first'],
                       help='Background removal strategy')
    
    # Add registry-specific commands
    parser.add_argument('--register-only', action='store_true', 
                       help='Only register files, don\'t process')
    parser.add_argument('--show-stats', action='store_true', 
                       help='Show current processing statistics')
    parser.add_argument('--reset-stalled', action='store_true',
                       help='Reset stalled files to pending status')
    parser.add_argument('--reset-failed', action='store_true',
                       help='Reset failed files to pending status')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = RegistryIntegratedExtractor(
        output_dir=args.output,
        registry_path=args.registry_db,
        debug=args.debug,
        enable_ai_analysis=not args.no_ai,
        bg_strategy=args.bg_strategy
    )
    
    # Handle registry-specific commands
    if args.show_stats:
        stats = extractor.registry.get_processing_stats()
        progress = extractor.registry.get_processing_progress()
        print(f"📊 Current Status: {stats}")
        print(f"📈 Progress: {progress}")
        return
    
    if args.reset_stalled:
        extractor.registry._reset_stalled_files()
        print("⚡ Reset stalled files to pending")
        return
    
    if args.reset_failed:
        extractor.registry.reset_failed_to_pending()
        return
    
    # Register input files
    registered_count = extractor.register_input_files(args.input)
    
    if args.register_only:
        print(f"📝 Registered {registered_count} files. Use without --register-only to process.")
        return
    
    if registered_count == 0:
        print("⚠️  No new files to register. All files may already be processed.")
        stats = extractor.registry.get_processing_stats() 
        if stats['pending'] > 0 or stats['processing'] > 0:
            print(f"📋 {stats['pending']} pending files available for processing")
        else:
            print("✅ No files pending processing")
            return
    
    # Run batch processing
    extractor.run_batch_processing()


if __name__ == "__main__":
    main()