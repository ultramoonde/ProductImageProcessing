#!/usr/bin/env python3
"""
Supabase-Integrated Image Processing Service
Enterprise-grade job management using existing Supabase infrastructure
"""

import argparse
import os
import cv2
import time
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging

# Import existing components
from tile_detector import TileDetector
from image_processor import ImageProcessor
from text_extractor import TextExtractor

# Import Supabase job service
from supabase_job_service import SupabaseJobService

# Import AI analysis if available
try:
    from local_consensus_analyzer import LocalConsensusAnalyzer
    AI_ANALYSIS_AVAILABLE = True
except ImportError:
    AI_ANALYSIS_AVAILABLE = False
    print("⚠️ AI consensus system not available - using basic OCR only")

class SupabaseImageProcessor:
    """Image processor service integrated with Supabase job management"""
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None,
                 enable_ai_analysis: bool = True, worker_name: str = None):
        """Initialize Supabase-integrated image processor"""
        
        # Initialize Supabase service
        self.job_service = SupabaseJobService(supabase_url, supabase_key)
        
        # Generate unique worker ID
        self.worker_id = str(uuid.uuid4())[:8]
        self.worker_name = worker_name or f"ImageProcessor-{self.worker_id}"
        
        # Register worker
        self.job_service.register_worker(self.worker_id, self.worker_name)
        
        # Initialize processing components
        self.tile_detector = TileDetector()
        self.image_processor = ImageProcessor(enable_background_removal=True)
        self.text_extractor = TextExtractor()
        
        # Initialize AI analysis
        self.enable_ai_analysis = enable_ai_analysis and AI_ANALYSIS_AVAILABLE
        if self.enable_ai_analysis:
            self.ai_analyzer = LocalConsensusAnalyzer(use_api_fallback=False)
            print("🏆 AI consensus system initialized")
        else:
            self.ai_analyzer = None
            print("📝 Using basic OCR analysis")
        
        print(f"🚀 Supabase processor initialized (Worker: {self.worker_id})")
    
    def create_processing_job(self, job_name: str, source_type: str, 
                            input_folder: str, processing_config: Dict = None,
                            created_by: str = None) -> str:
        """Create a new processing job and register all files"""
        
        # Find all image files
        input_path = Path(input_folder)
        file_paths = []
        
        if input_path.is_file():
            file_paths = [str(input_path)]
        else:
            extensions = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
            for ext in extensions:
                file_paths.extend([str(p) for p in input_path.glob(ext)])
        
        if not file_paths:
            raise ValueError(f"No image files found in {input_folder}")
        
        # Create job
        job_id = self.job_service.create_job(
            job_name=job_name,
            source_type=source_type,
            source_folder=str(input_folder),
            processing_config=processing_config or {
                'ai_analysis': self.enable_ai_analysis,
                'background_removal': True,
                'consensus_voting': True
            },
            created_by=created_by
        )
        
        # Register files
        registered_count = self.job_service.register_files_for_job(job_id, file_paths)
        
        print(f"✅ Created job '{job_name}' with {registered_count} files (ID: {job_id})")
        return job_id
    
    def process_available_work(self, max_files: int = None) -> Dict[str, int]:
        """Process available work from Supabase queue"""
        
        print(f"🔄 Worker {self.worker_id} starting processing...")
        
        stats = {
            'processed': 0,
            'failed': 0,
            'products_extracted': 0
        }
        
        processed_count = 0
        
        while True:
            # Check if we've hit max files limit
            if max_files and processed_count >= max_files:
                print(f"📊 Reached max files limit ({max_files})")
                break
            
            # Update heartbeat
            self.job_service.update_worker_heartbeat(self.worker_id)
            
            # Claim next file
            file_record = self.job_service.claim_next_file(self.worker_id)
            
            if not file_record:
                print("✅ No more files to process")
                break
            
            # Process the file
            success, results = self.process_single_file(file_record)
            
            if success:
                stats['processed'] += 1
                stats['products_extracted'] += results.get('products_extracted', 0)
                
                # Save products to database
                if results.get('products'):
                    self.job_service.save_extracted_products(
                        file_record['id'],
                        file_record['job_id'], 
                        results['products']
                    )
                
                # Mark file completed
                self.job_service.mark_file_completed(
                    file_record['id'], 
                    self.worker_id, 
                    results
                )
            else:
                stats['failed'] += 1
                error_message = results.get('error', 'Unknown error')
                self.job_service.mark_file_failed(
                    file_record['id'],
                    self.worker_id,
                    error_message
                )
            
            processed_count += 1
            
            # Progress report every 5 files
            if processed_count % 5 == 0:
                print(f"📈 Progress: {processed_count} files processed ({stats['processed']} success, {stats['failed']} failed)")
        
        print(f"🎉 Processing complete! Processed: {stats['processed']}, Failed: {stats['failed']}, Products: {stats['products_extracted']}")
        return stats
    
    def process_single_file(self, file_record: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Process a single file and return results"""
        
        file_path = file_record['file_path']
        
        try:
            start_time = time.time()
            print(f"📂 FILE PROCESSING START: {file_record['file_name']}")
            
            # File validation and analysis
            import os
            from pathlib import Path
            
            file_path_obj = Path(file_path)
            
            print(f"📋 FILE VALIDATION:")
            print(f"   📁 Path: {file_path}")
            print(f"   📄 Name: {file_record['file_name']}")
            print(f"   🔍 Exists: {file_path_obj.exists()}")
            
            if file_path_obj.exists():
                file_size_bytes = file_path_obj.stat().st_size
                file_size_mb = file_size_bytes / (1024 * 1024)
                file_extension = file_path_obj.suffix.lower()
                
                print(f"   📊 Size: {file_size_bytes:,} bytes ({file_size_mb:.2f}MB)")
                print(f"   🏷️  Extension: {file_extension}")
                print(f"   ✅ File validation: PASSED")
                
                # Size validation warnings
                if file_size_mb > 10:
                    print(f"   ⚠️  WARNING: Large file ({file_size_mb:.1f}MB) - may require more processing time")
                elif file_size_mb < 0.1:
                    print(f"   ⚠️  WARNING: Small file ({file_size_mb:.3f}MB) - may have quality issues")
            else:
                print(f"   ❌ File validation: FAILED - File does not exist")
                raise Exception(f"File not found: {file_path}")
            
            # Load and validate image
            print(f"🖼️  IMAGE LOADING:")
            print(f"   🔸 Loading image with OpenCV...")
            image = cv2.imread(file_path)
            
            if image is None:
                print(f"   ❌ OpenCV load failed - trying alternative methods...")
                # Try PIL as fallback
                try:
                    from PIL import Image as PILImage
                    import numpy as np
                    pil_image = PILImage.open(file_path)
                    image = np.array(pil_image)
                    # Convert RGB to BGR for OpenCV compatibility
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    print(f"   ✅ PIL fallback successful")
                except Exception as pil_error:
                    print(f"   ❌ PIL fallback failed: {pil_error}")
                    raise Exception(f"Could not load image with OpenCV or PIL: {file_path}")
            else:
                print(f"   ✅ OpenCV load successful")
            
            # Image properties analysis
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) > 2 else 1
            image_size_mb = (height * width * channels * image.dtype.itemsize) / (1024 * 1024)
            aspect_ratio = width / height
            
            print(f"📐 IMAGE PROPERTIES:")
            print(f"   📏 Dimensions: {width}x{height} (aspect: {aspect_ratio:.2f})")
            print(f"   🌈 Channels: {channels} ({'RGB' if channels == 3 else 'RGBA' if channels == 4 else 'Grayscale'})")
            print(f"   📊 Data type: {image.dtype}")
            print(f"   💾 Memory size: {image_size_mb:.2f}MB")
            
            # Image quality checks
            if width < 500 or height < 500:
                print(f"   ⚠️  WARNING: Low resolution image - may affect tile detection")
            if width > 5000 or height > 5000:
                print(f"   ⚠️  WARNING: High resolution image - processing may be slow")
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                print(f"   ⚠️  WARNING: Unusual aspect ratio - may affect processing")
            
            # Detect tiles
            print(f"🔍 TILE DETECTION:")
            print(f"   🔸 Running tile detector on {width}x{height} image...")
            tiles = self.tile_detector.detect_tiles(image)
            
            print(f"   ✅ Detected {len(tiles)} tiles in {file_record['file_name']}")
            
            if len(tiles) == 0:
                print(f"   ⚠️  WARNING: No tiles detected - check image content and detection parameters")
            elif len(tiles) > 50:
                print(f"   ⚠️  WARNING: Many tiles detected ({len(tiles)}) - processing may take time")
            
            if not tiles:
                raise Exception("No tiles detected in image")
            
            products = []
            
            # Process each tile
            for i, (x, y, w, h) in enumerate(tiles):
                try:
                    product_data = self.process_tile(
                        image, (x, y, w, h), i + 1,
                        file_record['file_name'], file_record['job_id']
                    )
                    
                    if product_data:
                        products.append(product_data)
                        
                except Exception as e:
                    print(f"⚠️ Failed to process tile {i+1}: {e}")
            
            # Calculate results
            processing_time = time.time() - start_time
            
            results = {
                'tiles_detected': len(tiles),
                'products_extracted': len(products),
                'processing_duration_seconds': processing_time,
                'products': products,
                'products_preview': products[:3]  # First 3 for quick preview
            }
            
            print(f"✅ Completed {file_record['file_name']}: {len(products)} products extracted")
            return True, results
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            print(f"❌ Failed {file_record['file_name']}: {error_msg}")
            return False, {'error': error_msg}
    
    def process_tile(self, image: any, tile_coords: Tuple[int, int, int, int], 
                    tile_num: int, source_file: str, job_id: str) -> Optional[Dict[str, Any]]:
        """Process individual tile and extract product data"""
        
        x, y, w, h = tile_coords
        source_h, source_w = image.shape[:2]
        source_channels = image.shape[2] if len(image.shape) > 2 else 1
        
        print(f"🔲 TILE EXTRACTION START:")
        print(f"   📁 Source file: {source_file}")
        print(f"   🔢 Tile number: {tile_num}")
        print(f"   🖼️  Source image: {source_w}x{source_h} (channels: {source_channels})")
        print(f"   📐 Extraction coordinates: x={x}, y={y}, w={w}, h={h}")
        print(f"   📊 Tile coverage: {(w*h)/(source_w*source_h)*100:.1f}% of source image")
        
        # Validate extraction coordinates
        valid_x = 0 <= x < source_w and 0 <= (x + w) <= source_w
        valid_y = 0 <= y < source_h and 0 <= (y + h) <= source_h
        print(f"   ✅ Coordinate validation: X={valid_x}, Y={valid_y}")
        
        if not (valid_x and valid_y):
            print(f"   ❌ EXTRACTION FAILED: Invalid coordinates!")
            return None
        
        # Extract tile image
        print(f"   🔸 Extracting tile region...")
        tile_image = image[y:y+h, x:x+w].copy()
        
        extracted_h, extracted_w = tile_image.shape[:2] 
        extracted_channels = tile_image.shape[2] if len(tile_image.shape) > 2 else 1
        extracted_size_mb = (extracted_h * extracted_w * extracted_channels * tile_image.dtype.itemsize) / (1024 * 1024)
        
        print(f"   ✅ Raw tile extracted: {extracted_w}x{extracted_h} ({extracted_size_mb:.3f}MB)")
        print(f"   📏 Expected vs actual: {w}x{h} → {extracted_w}x{extracted_h}")
        
        # Process tile (background removal, button removal, etc.)
        print(f"   🔸 Starting image processor pipeline...")
        processed_tile, text_region = self.image_processor.process_tile(tile_image)
        print(f"   🔸 Image processor pipeline completed")
        
        if processed_tile is None:
            print(f"   ❌ PROCESSING FAILED: Image processor returned None!")
            return None
        
        # Save processed tile for visual dashboard
        tile_filename = f"{source_file}_tile_{tile_num}.png"
        tile_path = f"static/tiles/{tile_filename}"
        
        final_h, final_w = processed_tile.shape[:2]
        final_channels = processed_tile.shape[2] if len(processed_tile.shape) > 2 else 1
        final_size_mb = (final_h * final_w * final_channels * processed_tile.dtype.itemsize) / (1024 * 1024)
        
        print(f"   🔸 TILE SAVE OPERATION:")
        print(f"      📁 Output path: {tile_path}")
        print(f"      📐 Final tile: {final_w}x{final_h} (channels: {final_channels})")
        print(f"      💾 Final size: {final_size_mb:.3f}MB")
        print(f"      🎯 Background removal: ENABLED (remove_background=True)")
        
        if self.image_processor.save_tile_as_png(processed_tile, tile_path, remove_background=True):
            print(f"      ✅ SAVE SUCCESS: {tile_filename}")
        else:
            print(f"      ❌ SAVE FAILED: {tile_filename}")
            tile_filename = None
        
        # Extract OCR text
        ocr_input = text_region if text_region is not None else processed_tile
        ocr_h, ocr_w = ocr_input.shape[:2]
        ocr_channels = ocr_input.shape[2] if len(ocr_input.shape) > 2 else 1
        
        print(f"   🔸 OCR TEXT EXTRACTION:")
        print(f"      📐 OCR input: {ocr_w}x{ocr_h} (channels: {ocr_channels})")
        print(f"      🎯 Using: {'text_region' if text_region is not None else 'processed_tile'}")
        
        ocr_text = self.text_extractor.extract_text_from_tile(ocr_input)
        ocr_length = len(ocr_text.strip()) if ocr_text else 0
        
        print(f"      ✅ OCR result: '{ocr_text[:50]}{'...' if len(ocr_text) > 50 else ''}' ({ocr_length} chars)")
        
        # Initialize product data
        print(f"   🔸 PRODUCT DATA INITIALIZATION:")
        product_data = {
            'source_file': source_file,
            'tile_position': f"x={x}, y={y}, w={w}, h={h}",
            'ocr_text_raw': ocr_text,
            'extraction_method': 'AI_Consensus' if self.enable_ai_analysis else 'OCR_Basic',
            'worker_id': self.worker_id,
            'extracted_image_filename': tile_filename  # Visual tile for dashboard
        }
        
        print(f"      📊 Method: {product_data['extraction_method']}")
        print(f"      🆔 Worker: {self.worker_id}")
        print(f"      🖼️  Dashboard tile: {tile_filename}")
        
        # AI analysis if available
        if self.enable_ai_analysis and self.ai_analyzer:
            try:
                print(f"   🔸 AI CONSENSUS ANALYSIS START:")
                print(f"      🤖 AI analyzer available: True")
                print(f"      📐 Sending processed_tile: {final_w}x{final_h}")
                print(f"      📐 Sending text_region: {text_region.shape[:2] if text_region is not None else 'None'}")
                
                ai_result = asyncio.run(self.ai_analyzer.analyze_product_with_consensus(processed_tile, text_region))
                
                print(f"      ✅ AI analysis completed")
                print(f"      📋 AI result keys: {list(ai_result.keys()) if ai_result else 'None'}")
                
                if ai_result:
                    product_name = ai_result.get('product_name', '')
                    confidence = ai_result.get('confidence', 0)
                    print(f"      🏷️  Product: '{product_name}' (confidence: {confidence})")
                    print(f"      🏢 Brand: '{ai_result.get('brand', '')}'")
                    print(f"      💰 Price: '{ai_result.get('price', '')}'")
                    print(f"      ⚖️  Weight: '{ai_result.get('weight', '')}'")
                else:
                    print(f"      ⚠️  AI result is empty or None")
                
                # Merge AI results
                product_data.update({
                    'product_name': ai_result.get('product_name', ''),
                    'brand': ai_result.get('brand', ''),
                    'manufacturer': ai_result.get('manufacturer', ''),
                    'price': ai_result.get('price', ''),
                    'weight': ai_result.get('weight', ''),
                    'quantity': ai_result.get('quantity', ''),
                    'ai_confidence_score': ai_result.get('confidence', 0),
                    'processing_metadata': {
                        'consensus_info': ai_result.get('consensus_info', ''),
                        'model_responses': ai_result.get('model_responses', [])
                    }
                })
                
                # Parse numeric values
                print(f"      🔸 Parsing numeric data from AI results...")
                product_data.update(self._parse_numeric_data(product_data))
                print(f"      ✅ AI analysis pipeline completed successfully")
                
            except Exception as e:
                print(f"      ❌ AI ANALYSIS FAILED: {str(e)}")
                print(f"      🔄 Falling back to OCR-only processing...")
                product_data['extraction_errors'] = str(e)
        else:
            print(f"   🔸 AI FALLBACK - OCR BASIC PARSING:")
            if not self.enable_ai_analysis:
                print(f"      ℹ️  AI analysis disabled")
            elif not self.ai_analyzer:
                print(f"      ⚠️  AI analyzer not available")
            
            # Basic OCR parsing
            print(f"      🔸 Running basic OCR text parsing...")
            parsed_data = self._parse_ocr_basic(ocr_text)
            product_data.update(parsed_data)
            
            if parsed_data:
                parsed_keys = list(parsed_data.keys())
                print(f"      ✅ OCR parsing completed: {len(parsed_keys)} fields extracted")
                print(f"      📋 Extracted fields: {parsed_keys}")
            else:
                print(f"      ⚠️  OCR parsing returned no data")
        
        # Calculate data quality score
        print(f"   🔸 QUALITY ASSESSMENT:")
        product_data['data_quality_score'] = self._calculate_quality_score(product_data)
        product_data['manual_review_required'] = product_data['data_quality_score'] < 60
        
        print(f"      📊 Quality score: {product_data['data_quality_score']}/100")
        print(f"      🔍 Manual review: {'REQUIRED' if product_data['manual_review_required'] else 'NOT NEEDED'}")
        
        print(f"🎯 TILE PROCESSING COMPLETE:")
        print(f"   📦 Product: '{product_data.get('product_name', 'UNKNOWN')}'")
        print(f"   🏢 Brand: '{product_data.get('brand', 'UNKNOWN')}'")
        print(f"   📊 Quality: {product_data['data_quality_score']}/100")
        print(f"   🔬 Method: {product_data['extraction_method']}")
        print("")
        
        return product_data
    
    def _parse_numeric_data(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse numeric values from text fields"""
        import re
        
        updates = {}
        
        # Parse price
        price_str = product_data.get('price', '')
        if price_str:
            price_match = re.search(r'(\d+[,.]?\d*)', price_str.replace(',', '.'))
            if price_match:
                try:
                    updates['price_numeric'] = float(price_match.group(1))
                    updates['currency'] = 'EUR' if '€' in price_str else 'USD' if '$' in price_str else ''
                except ValueError:
                    pass
        
        # Parse weight
        weight_str = product_data.get('weight', '')
        if weight_str:
            weight_match = re.search(r'(\d+[,.]?\d*)\s*(g|kg|ml|l)', weight_str.lower())
            if weight_match:
                try:
                    updates['weight_numeric'] = float(weight_match.group(1).replace(',', '.'))
                    updates['weight_unit'] = weight_match.group(2)
                except ValueError:
                    pass
        
        # Check for organic/bio indicators
        product_name = product_data.get('product_name', '').lower()
        brand = product_data.get('brand', '').lower()
        updates['organic_certified'] = any(keyword in product_name or keyword in brand 
                                         for keyword in ['bio', 'organic', 'öko'])
        
        return updates
    
    def _parse_ocr_basic(self, ocr_text) -> Dict[str, str]:
        """Basic OCR text parsing when AI is not available"""
        import re
        
        # Handle both dict and string input
        if isinstance(ocr_text, dict):
            # If it's already a dict from extract_text_from_tile(), use it directly
            result = {
                'product_name': ocr_text.get('product_name', ''),
                'brand': ocr_text.get('manufacturer', ''),  # Map manufacturer to brand
                'price': ocr_text.get('price', ''),
                'weight': ocr_text.get('weight', ''),
                'price_per_unit': ocr_text.get('price_per_unit', ''),
                'cost_per_unit': ocr_text.get('cost_per_unit', ''),
                'discount': ocr_text.get('discount', ''),
                'additional_info': ocr_text.get('additional_info', '')
            }
            return {k: v for k, v in result.items() if v}  # Remove empty values
        
        # Handle string input (legacy/fallback)
        result = {}
        if not ocr_text:
            return result
        
        lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
        
        # Extract price
        for line in lines:
            price_match = re.search(r'(\d+[,.]?\d*)\s*€', line)
            if price_match and not result.get('price'):
                result['price'] = f"{price_match.group(1)}€"
                break
        
        # Extract product name (usually first meaningful line)
        if lines:
            for line in lines:
                if not re.search(r'^\d+[,.]?\d*\s*€', line) and len(line) > 3:
                    result['product_name'] = line
                    break
        
        return result
    
    def _calculate_quality_score(self, product_data: Dict[str, Any]) -> float:
        """Calculate data quality score based on field completeness"""
        essential_fields = ['product_name', 'price', 'brand']
        filled_essential = sum(1 for field in essential_fields if product_data.get(field))
        
        all_fields = ['product_name', 'price', 'brand', 'weight', 'quantity', 'manufacturer']
        filled_all = sum(1 for field in all_fields if product_data.get(field))
        
        essential_score = (filled_essential / len(essential_fields)) * 100
        completeness_score = (filled_all / len(all_fields)) * 100
        
        return round(essential_score * 0.7 + completeness_score * 0.3, 2)
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job"""
        return self.job_service.get_job_status(job_id)
    
    def list_active_jobs(self) -> List[Dict[str, Any]]:
        """List all active jobs"""
        return self.job_service.get_active_jobs()
    
    def export_job_to_csv(self, job_id: str, output_path: str) -> bool:
        """Export job results to CSV"""
        return self.job_service.export_job_results_to_csv(job_id, output_path)


def main():
    parser = argparse.ArgumentParser(description='Supabase-Integrated Image Processing Service')
    
    # Connection settings
    parser.add_argument('--supabase-url', help='Supabase project URL')
    parser.add_argument('--supabase-key', help='Supabase anon key')
    parser.add_argument('--worker-name', help='Custom worker name')
    
    # Job creation
    parser.add_argument('--create-job', help='Create new job with given name')
    parser.add_argument('--source-type', default='flink', help='Source type (flink, dr_morris, custom)')
    parser.add_argument('-i', '--input', help='Input folder path')
    parser.add_argument('--created-by', help='User ID who created the job')
    
    # Processing  
    parser.add_argument('--process', action='store_true', help='Process available work')
    parser.add_argument('--max-files', type=int, help='Maximum files to process')
    
    # Status and management
    parser.add_argument('--list-jobs', action='store_true', help='List active jobs')
    parser.add_argument('--job-status', help='Get status of specific job ID')
    parser.add_argument('--export-csv', help='Export job results to CSV (requires --job-id)')
    parser.add_argument('--job-id', help='Job ID for operations')
    parser.add_argument('-o', '--output', help='Output file path')
    
    # Processing options
    parser.add_argument('--no-ai', action='store_true', help='Disable AI analysis')
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = SupabaseImageProcessor(
            supabase_url=args.supabase_url,
            supabase_key=args.supabase_key,
            enable_ai_analysis=not args.no_ai,
            worker_name=args.worker_name
        )
        
        # Handle different operations
        if args.create_job:
            if not args.input:
                parser.error("--input required when creating job")
            
            job_id = processor.create_processing_job(
                job_name=args.create_job,
                source_type=args.source_type,
                input_folder=args.input,
                created_by=args.created_by
            )
            print(f"📝 Job created with ID: {job_id}")
            
            # Optionally start processing immediately
            if args.process:
                print("🚀 Starting processing...")
                processor.process_available_work(args.max_files)
        
        elif args.process:
            stats = processor.process_available_work(args.max_files)
            print(f"📊 Final stats: {stats}")
        
        elif args.list_jobs:
            jobs = processor.list_active_jobs()
            print(f"📋 Active jobs ({len(jobs)}):")
            for job in jobs:
                print(f"  • {job['job_name']} ({job['id'][:8]}): {job['status']} - {job['completion_percentage']:.1f}% complete")
        
        elif args.job_status:
            status = processor.get_job_status(args.job_status)
            if status:
                print(f"📊 Job Status: {status['job_name']}")
                print(f"   Status: {status['status']}")
                print(f"   Progress: {status['completion_percentage']:.1f}%")
                print(f"   Files: {status['files_completed']}/{status['total_files']} completed")
                print(f"   Products: {status['total_products_extracted']} extracted")
            else:
                print(f"❌ Job {args.job_status} not found")
        
        elif args.export_csv:
            if not args.job_id or not args.output:
                parser.error("--job-id and --output required for CSV export")
            
            success = processor.export_job_to_csv(args.job_id, args.output)
            if success:
                print(f"📊 Exported job {args.job_id} to {args.output}")
            else:
                print(f"❌ Failed to export job {args.job_id}")
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()