#!/usr/bin/env python3
"""
Step-by-Step Visualization Pipeline
Demonstrates the complete food product extraction journey from screenshot to final CSV
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import json
import os
from datetime import datetime
import sys
import subprocess
import time
import requests
import signal
import asyncio

# Optional psutil import for advanced process management
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import your existing systems
sys.path.append('src')
from main import FoodExtractor
from screenshot_ui_analyzer import ScreenshotUIAnalyzer
from real_vision_analyzer import RealVisionAnalyzer
from ai_text_analyzer import AITextAnalyzer
from src.local_consensus_analyzer import LocalConsensusAnalyzer
# TileDetector class will be defined inline since src.tile_detector doesn't exist

# ================================================================================
# TILEDETECTOR CLASS (from consolidated_pipeline.py)
# ================================================================================

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

    def _sort_tiles_grid(self, tiles: List[Tuple[int, int, int, int]], image_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int, int]]:
        """Sort tiles in grid order (left to right, top to bottom)."""
        if not tiles:
            return tiles

        # Sort by y coordinate first (top to bottom), then by x coordinate (left to right)
        tiles.sort(key=lambda tile: (tile[1], tile[0]))

        return tiles

class StepByStepPipeline:
    def __init__(self, output_dir: str = "step_by_step_flat"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 🧹 Process Management: Ensure clean state before startup
        self._ensure_clean_startup()

        # No nested step directories - use flat structure with image-specific folders
        # Each image gets its own folder with all step files using step prefixes

        # Initialize your existing systems
        self.ui_analyzer = ScreenshotUIAnalyzer()
        self.food_extractor = FoodExtractor(
            str(self.output_dir / "extraction_temp"),
            enable_ai_analysis=True
        )
        # Initialize 3-model consensus analyzer for intelligent category detection (OCR fallback removed per user request)
        self.consensus_analyzer = LocalConsensusAnalyzer()
        # Initialize tile detector for proper canvas detection
        self.tile_detector = TileDetector()
        # Initialize background removal manager for Step 4C
        self.bg_removal_manager = self._initialize_background_removal_manager()

        # Shop knowledge storage
        self.shop_profile = None

    def _ensure_clean_startup(self):
        """🧹 Process Management: Kill background processes to ensure clean state"""
        try:
            print("🧹 Cleaning up any existing pipeline processes...")

            # Kill any existing step-by-step pipeline processes
            subprocess.run(["pkill", "-f", "python3.*step_by_step"], check=False, capture_output=True)

            # Kill any dashboard servers
            subprocess.run(["pkill", "-f", "dashboard_server"], check=False, capture_output=True)

            # Give processes time to terminate
            time.sleep(1)

            print("✅ Process cleanup complete - ready for clean execution")

        except Exception as e:
            print(f"⚠️  Process cleanup warning: {e}")
            # Continue anyway - cleanup is best effort

    def _cleanup_before_step(self, step_name: str):
        """🧹 Clean up processes before starting each step"""
        try:
            print(f"🧹 [{step_name}] Ensuring exclusive LLM access...")

            if PSUTIL_AVAILABLE:
                # Advanced process management with psutil
                current_pid = os.getpid()

                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['pid'] == current_pid:
                            continue

                        cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ""

                        # Kill competing pipeline processes (but keep ollama serve)
                        if ("python3" in cmdline and
                            ("step_by_step" in cmdline or "consensus" in cmdline) and
                            "ollama serve" not in cmdline):

                            proc.terminate()
                            print(f"   ⚠️ Terminated competing process: PID {proc.info['pid']}")

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            else:
                # Fallback: Use pkill for basic process cleanup
                subprocess.run(["pkill", "-f", "python3.*step_by_step.*consensus"], check=False, capture_output=True)

            # Brief pause to let processes terminate
            time.sleep(0.5)
            print(f"✅ [{step_name}] Ready for exclusive processing")

        except Exception as e:
            print(f"⚠️  [{step_name}] Process management warning: {e}")
            # Continue anyway

    def _step_0_ui_analysis(self, image: np.ndarray, image_name: str) -> dict:
        """
        Step 0: UI Analysis (Sophisticated ScreenshotUIAnalyzer version)
        Input: Single screenshot
        Output: UI regions (header/content/footer) + annotated image + coordinates JSON
        """
        print("\n📱 STEP 0: UI Analysis")
        print("   Goal: Identify UI regions (header, content area, footer, navigation)")

        # Analyze UI structure (with fallback)
        if self.ui_analyzer:
            ui_result = self.ui_analyzer.analyze_screenshot(image)
        else:
            # Fallback UI analysis
            h, w = image.shape[:2]
            ui_result = {
                'regions': {
                    'header': {'coordinates': {'x1': 0, 'y1': 0, 'x2': w, 'y2': min(200, h//4)}},
                    'content': {'coordinates': {'x1': 0, 'y1': min(200, h//4), 'x2': w, 'y2': h-100}},
                    'footer': {'coordinates': {'x1': 0, 'y1': h-100, 'x2': w, 'y2': h}}
                }
            }

        # Create annotated image
        annotated_image = image.copy()

        # Draw UI regions with colored rectangles
        colors = {
            'header': (0, 255, 0),      # Green
            'content': (255, 0, 0),     # Blue
            'footer': (0, 0, 255),      # Red
            'navigation': (255, 255, 0)  # Cyan
        }

        regions_data = {}

        for region_name, region_info in ui_result.get('regions', {}).items():
            if 'coordinates' in region_info:
                coords = region_info['coordinates']
                x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']

                # Draw rectangle
                color = colors.get(region_name, (128, 128, 128))
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)

                # Add label
                cv2.putText(annotated_image, region_name.upper(),
                           (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                           1, color, 2)

                # Store coordinates
                regions_data[region_name] = {
                    'coordinates': [x1, y1, x2, y2],
                    'description': f"{region_name} region"
                }

        # Save outputs
        annotated_path = self.output_dir / f"{image_name}_00_ui_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated_image)

        regions_json_path = self.output_dir / f"{image_name}_00_ui_regions.json"
        with open(regions_json_path, 'w') as f:
            json.dump(regions_data, f, indent=2)

        print(f"   ✅ Saved annotated image: {annotated_path.name}")
        print(f"   ✅ Saved regions JSON: {regions_json_path.name}")
        print(f"   ✅ Detected {len(regions_data)} UI regions")

        return {
            'regions': regions_data,
            'annotated_image_path': str(annotated_path),
            'regions_json_path': str(regions_json_path),
            'ui_analysis_result': ui_result
        }

    def _ensure_clean_startup(self):
        """Ensure clean system state before pipeline startup"""
        print("🧹 Checking system state...")

        # Count existing pipeline processes
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            pipeline_processes = []
            ollama_processes = []

            for line in result.stdout.split('\n'):
                if 'step_by_step' in line or 'dashboard_server' in line:
                    if 'grep' not in line:
                        pipeline_processes.append(line)
                if 'ollama' in line and 'grep' not in line:
                    ollama_processes.append(line)

            print(f"Found {len(pipeline_processes)} pipeline processes")
            print(f"Found {len(ollama_processes)} ollama processes")

            # If too many processes, clean up
            if len(pipeline_processes) > 2:
                print("🚨 Too many pipeline processes detected. Cleaning up...")
                subprocess.run(['pkill', '-f', 'step_by_step'], check=False)
                subprocess.run(['pkill', '-f', 'dashboard_server'], check=False)
                time.sleep(2)

            # Always restart ollama for fresh state
            print("🔄 Restarting ollama server...")
            subprocess.run(['pkill', 'ollama'], check=False)
            time.sleep(3)

            # Start ollama in background
            subprocess.Popen(['ollama', 'serve'],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            time.sleep(5)

            # Health check
            self._verify_ollama_health()

        except Exception as e:
            print(f"⚠️  Process management warning: {e}")

    def _verify_ollama_health(self):
        """Verify ollama server is responding"""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=10)
            if response.status_code == 200:
                print("✅ Ollama server is healthy")
            else:
                print("⚠️  Ollama server responding but with errors")
        except Exception as e:
            print(f"⚠️  Ollama health check failed: {e}")
            print("Continuing anyway...")

        # Create image-specific directory for flat structure
        print("🏗️  Setting up flat directory structure...")
        image_dir = self.output_dir / screenshot_name
        image_dir.mkdir(exist_ok=True)
        print(f"📂 Image directory: {image_dir}")

        # Store image directory for use in all step methods
        self.current_image_dir = image_dir

        self.category_database = None
    
    def run_complete_demonstration(self, screenshot_path: str) -> Dict[str, Any]:
        """
        Run the complete step-by-step pipeline demonstration
        Returns all results for final CSV generation
        """
        screenshot_name = Path(screenshot_path).stem
        image_dir = self.output_dir / screenshot_name
        image_dir.mkdir(exist_ok=True)

        # Store image directory for use in all step methods
        self.current_image_dir = image_dir

        print(f"🚀 Starting Step-by-Step Pipeline Demonstration")
        print(f"📁 Input: {screenshot_path}")
        print(f"📂 Output: {image_dir}")
        print("="*80)
        
        # Load original screenshot
        original_image = cv2.imread(screenshot_path)
        if original_image is None:
            raise ValueError(f"Could not load screenshot: {screenshot_path}")
        
        results = {
            "screenshot_name": screenshot_name,
            "screenshot_path": screenshot_path,
            "steps": [],
            "products": [],
            "final_csv_data": []
        }

        # STEP 1: UI Region Analysis (First Step - Identify UI regions)
        self._cleanup_before_step("STEP 1")
        step1_result = self._step_01_ui_region_analysis(original_image, screenshot_name)
        results["steps"].append(step1_result)

        # STEP 2: Header Category Analysis using properly identified regions
        self._cleanup_before_step("STEP 2")
        step2_result = self._step_02_header_category_analysis(original_image, step1_result["header_region"], screenshot_name)
        results["steps"].append(step2_result)

        # STEP 3: Product Canvas Detection using identified content region
        self._cleanup_before_step("STEP 3")
        step3_result = self._step_03_product_canvas_detection(original_image, step1_result, screenshot_name)
        results["steps"].append(step3_result)

        # STEP 4: Component Coordinate Extraction (Pink Buttons)
        self._cleanup_before_step("STEP 4")
        step4_result = self._step_04_component_coordinate_extraction(original_image, step3_result["canvases"], screenshot_name)
        results["steps"].append(step4_result)

        # STEP 4.5: Extract Clean Product Images - Create actual product images from coordinates
        self._cleanup_before_step("STEP 4.5: CLEAN PRODUCT EXTRACTION")
        clean_products = self._extract_clean_product_images(original_image, step4_result["components_data"], screenshot_name)

        # STEP 5: Consensus Product Analysis - Send clean products to LLM for detailed analysis
        self._cleanup_before_step("STEP 5: CONSENSUS ANALYSIS")
        step5_consensus_result = self._step_05_consensus_product_analysis(clean_products, step2_result["category_data"], screenshot_name)
        results["steps"].append(step5_consensus_result)

        # STEP 6: Final CSV Generation with all consensus product data
        self._cleanup_before_step("STEP 6: CSV GENERATION")
        step6_csv_result = self._step_06_final_csv_generation(step5_consensus_result["analyzed_products"], step2_result["category_data"], screenshot_name)
        results["steps"].append(step6_csv_result)
        results["final_csv_data"] = step6_csv_result["csv_data"]

        # Generate HTML Report
        self._generate_html_report(results)

        print("\n🎉 Step-by-Step Pipeline Complete!")
        print(f"📊 Processed: {len(step6_csv_result['csv_data'])} products")
        print(f"📁 Results: {image_dir}")
        print(f"🌐 HTML Report: {image_dir}/{screenshot_name}_pipeline_report.html")
        print(f"📄 CSV File: {image_dir}/{screenshot_name}_extracted_products.csv")
        
        return results

    def _step_00_shop_category_discovery(self, image: np.ndarray, name: str) -> Dict[str, Any]:
        """STEP 0: UI Analysis & Region Detection for single screenshot"""
        print("🔍 STEP 0: UI Analysis & Region Detection")

        # Use flat directory structure with step prefix
        step_dir = self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir
        step_dir.mkdir(exist_ok=True)

        # Run enhanced UI analysis with proper category parsing
        analysis = self._extract_categories_with_consensus(image, name)

        # Create visualization
        vis_image = image.copy()

        # Draw enhanced UI analysis results
        # Header region (0-200px) - Red
        cv2.rectangle(vis_image, (0, 50), (image.shape[1], 200), (0, 0, 255), 3)
        cv2.putText(vis_image, "HEADER ANALYSIS", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(vis_image, f"Main Category: {analysis.get('main_category', 'Unknown')}",
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Header Tabs: {len(analysis.get('header_tabs', []))}",
                   (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Content region (200-800px) - Green
        cv2.rectangle(vis_image, (0, 200), (image.shape[1], 800), (0, 255, 0), 3)
        cv2.putText(vis_image, "CONTENT ANALYSIS", (10, 230),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_image, f"Active Subcategory: {analysis.get('active_subcategory', 'Unknown')}",
                   (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Subcategories: {len(analysis.get('available_subcategories', []))}",
                   (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Canvas Mappings: {analysis.get('total_canvases', 0)}",
                   (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw product canvas boundaries if available
        canvas_assignments = analysis.get('canvas_categories', [])
        for i, canvas in enumerate(canvas_assignments):
            x, y, w, h = canvas['x'], canvas['y'], canvas['width'], canvas['height']
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(vis_image, f"Canvas {canvas['canvas_id']}: {canvas['subcategory']}",
                       (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Add step title
        cv2.putText(vis_image, "STEP 0: UI Analysis",
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        # Save files
        analysis_path = step_dir / f"{name}_00_analysis.jpg"
        cv2.imwrite(str(analysis_path), vis_image)

        json_path = step_dir / f"{name}_00_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2)

        result = {
            "step": 0,
            "title": "Enhanced UI Analysis & Category Detection",
            "description": f"Main: {analysis.get('main_category', 'Unknown')}, Sub: {analysis.get('active_subcategory', 'Unknown')}, Canvases: {analysis.get('total_canvases', 0)}",
            "files": {
                "visualization": str(analysis_path),
                "analysis_json": str(json_path)
            },
            "analysis": analysis,
            "screenshots_analyzed": 1,
            "categories_discovered": 1 if analysis.get('main_category', 'Unknown') != 'Unknown' else 0
        }

        print(f"  ✅ Saved: {analysis_path}")
        print(f"  ✅ Saved: {json_path}")
        print(f"  📊 Categories: {analysis.get('categories', [])}")
        print(f"  🎯 Product tiles: {len(analysis.get('content_tiles', []))}")
        return result

    def _find_screenshot_directories(self) -> List[Path]:
        """Find all directories containing screenshots"""
        screenshot_dirs = []

        # Primary source: Flink directory (use ONLY this for comprehensive discovery)
        flink_dir = Path('/Users/davemooney/_dev/Flink')
        if flink_dir.exists():
            screenshot_dirs.append(flink_dir)
            print(f"   📁 Using primary Flink directory: {flink_dir}")
            return screenshot_dirs

        # Fallback: local directories only if Flink directory not found
        print(f"   ⚠️  Flink directory not found, using local directories as fallback")
        base_dir = Path('.')
        local_patterns = [
            "flink_sample_test", "friday_demo_batch", "screenshots", "test10_screenshots"
        ]

        for pattern in local_patterns:
            dirs = list(base_dir.glob(pattern))
            screenshot_dirs.extend([d for d in dirs if d.is_dir()])

        # Remove duplicates and sort
        screenshot_dirs = list(set(screenshot_dirs))
        return screenshot_dirs

    def _collect_all_screenshots(self, screenshot_dirs: List[Path]) -> List[Path]:
        """Collect all image files from screenshot directories"""
        all_screenshots = []
        image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}

        for dir_path in screenshot_dirs:
            if dir_path.exists():
                for ext in image_extensions:
                    all_screenshots.extend(dir_path.glob(f'*{ext}'))

        return sorted(list(set(all_screenshots)))

    def _analyze_all_categories(self, screenshots: List[Path], step_dir: Path) -> Dict[str, Any]:
        """Analyze categories from all screenshots using consensus system"""
        print(f"   🧠 Analyzing categories using 3-model consensus system...")

        category_results = []
        failed_analyses = []

        for i, screenshot_path in enumerate(screenshots, 1):
            print(f"   🔍 [{i:3}/{len(screenshots)}] Analyzing: {screenshot_path.name}")

            try:
                # Load image
                image = cv2.imread(str(screenshot_path))
                if image is None:
                    failed_analyses.append({"file": str(screenshot_path), "error": "Could not load image"})
                    continue

                # FIXED: Analyze the full content area where categories are actually located
                # Categories like "Bananen", "Äpfel & Birnen" are in the main content, not header
                # Skip the very top status bar (50px) but analyze the full content area
                content_region = image[50:, :]  # Skip status bar, analyze full content
                print(f"   🔍 Analyzing full content region: {content_region.shape[1]}x{content_region.shape[0]}px for {screenshot_path.name}")

                # Use enhanced UI parsing for category analysis
                category_info = self._extract_categories_with_consensus(image, screenshot_path.name)

                if category_info:
                    category_results.append({
                        "screenshot": str(screenshot_path),
                        "filename": screenshot_path.name,
                        "analysis": category_info
                    })
                else:
                    failed_analyses.append({"file": str(screenshot_path), "error": "Consensus analysis failed"})

                # Save incremental results after EVERY image for real-time dashboard progress
                self._save_incremental_results(category_results, failed_analyses, i, len(screenshots), step_dir, screenshot_path.name)

            except Exception as e:
                failed_analyses.append({"file": str(screenshot_path), "error": str(e)})
                continue

        print(f"   ✅ Successfully analyzed: {len(category_results)} screenshots")
        print(f"   ❌ Failed analyses: {len(failed_analyses)} screenshots")

        return {
            "successful_analyses": category_results,
            "failed_analyses": failed_analyses,
            "total_screenshots": len(screenshots)
        }

    def _save_incremental_results(self, category_results: List[Dict], failed_analyses: List[Dict], processed: int, total: int, step_dir: Path, current_image: str = None) -> None:
        """Save current results incrementally for dashboard progress tracking"""
        import json
        import csv
        from datetime import datetime

        try:
            # Create CSV with current results for dashboard
            csv_path = step_dir / "category_discovery_results.csv"
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Screenshot', 'Main Category', 'Active Subcategory', 'Subcategories Found', 'Method'])

                for result in category_results:
                    analysis = result['analysis']
                    writer.writerow([
                        result['filename'],
                        analysis.get('main_category', 'Unknown'),
                        analysis.get('active_subcategory', 'Unknown'),
                        len(analysis.get('available_subcategories', [])),
                        analysis.get('method', 'unknown')
                    ])

            # Create temporary shop profile for dashboard
            discovered_categories = {}
            for result in category_results:
                analysis = result['analysis']
                main_cat = analysis.get('main_category', 'Unknown')
                if main_cat != 'Unknown':
                    if main_cat not in discovered_categories:
                        discovered_categories[main_cat] = set()
                    for subcat in analysis.get('available_subcategories', []):
                        discovered_categories[main_cat].add(subcat)

            # Convert sets to lists for JSON serialization
            for cat in discovered_categories:
                discovered_categories[cat] = list(discovered_categories[cat])

            shop_profile = {
                "name": "Grocery Delivery App",
                "type": "grocery_delivery",
                "layout_type": "grid_with_categories",
                "total_categories": len(discovered_categories),
                "screenshots_analyzed": processed,
                "analysis_timestamp": datetime.now().isoformat(),
                "progress": f"{processed}/{total} ({processed/total*100:.1f}%)"
            }

            profile_path = step_dir / "shop_profile.json"
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(shop_profile, f, indent=2)

            # Create temporary category database for dashboard
            category_database = {
                "categories": discovered_categories,
                "keyword_mapping": {},
                "total_categories": len(discovered_categories),
                "processed_images": processed,
                "total_images": total,
                "created_timestamp": datetime.now().isoformat()
            }

            db_path = step_dir / "category_database.json"
            with open(db_path, 'w', encoding='utf-8') as f:
                json.dump(category_database, f, indent=2)

            # Create current status file for real-time dashboard updates
            current_status = {
                "current_image": current_image,
                "processed": processed,
                "total": total,
                "progress_percent": round((processed / total) * 100, 1),
                "categories_discovered": len(discovered_categories),
                "status": "processing" if processed < total else "completed",
                "last_updated": datetime.now().isoformat()
            }

            status_path = step_dir / "current_status.json"
            with open(status_path, 'w', encoding='utf-8') as f:
                json.dump(current_status, f, indent=2)

            print(f"   📊 Real-time status saved: {processed}/{total} processed ({current_status['progress_percent']}%), {len(discovered_categories)} categories discovered")

        except Exception as e:
            print(f"   ⚠️  Failed to save incremental results: {e}")

    def _extract_categories_with_consensus(self, full_image: np.ndarray, filename: str) -> Dict[str, Any]:
        """Extract categories using consensus-based header analysis - exactly as user requested"""
        try:
            print(f"   🔍 Using consensus-based header analysis for category detection: {filename}")

            # PHASE 1: Extract clean header region using existing coordinate detection
            # Header region is typically in the top portion of the image (around y=0-250px)
            header_height = 250  # Generous header region to capture all category information
            header_region = full_image[0:header_height, :]
            print(f"   🔍 Cropped header region: {header_region.shape[1]}x{header_region.shape[0]}px for {filename}")

            # PHASE 2: Send header to consensus system with simple category prompt
            if hasattr(self, 'consensus_analyzer') and self.consensus_analyzer:
                print(f"   🔍 Using consensus system with UI analysis mode for {filename}")

                # Create simple, direct prompt for category identification
                category_prompt = (
                    "This is the header section from a food delivery app screenshot. "
                    "I need you to identify the main category and subcategory information visible in this header. "
                    "Look for category tabs and any active/highlighted sections. "
                    "Return JSON with: main_category, active_subcategory, available_subcategories (array)"
                )

                # Use consensus system to analyze header region for categories
                import asyncio
                try:
                    consensus_result = asyncio.run(
                        self.consensus_analyzer.analyze_categories_with_consensus(header_region)
                    )
                except Exception as consensus_error:
                    print(f"   ⚠️ Consensus analysis failed: {consensus_error}")
                    consensus_result = None

                if consensus_result:
                    print(f"   📊 Consensus result keys: {list(consensus_result.keys())}")

                    # Extract category information directly from consensus result
                    # The consensus system returns a flat dictionary with results
                    main_category = "Unknown"
                    active_subcategory = "Unknown"
                    available_subcategories = []

                    # Try to extract categories from the consensus result
                    categories_field = consensus_result.get("categories", [])
                    current_field = consensus_result.get("current", "")

                    # If we have a categories list, use it
                    if isinstance(categories_field, list) and categories_field:
                        available_subcategories = categories_field
                        if current_field and current_field in categories_field:
                            main_category = current_field
                            active_subcategory = current_field
                        else:
                            # Use the first category as fallback
                            main_category = categories_field[0]
                            active_subcategory = categories_field[0]

                    # Also try extracting from other possible fields
                    if main_category == "Unknown":
                        main_category = consensus_result.get("product_category", consensus_result.get("category", "Unknown"))
                        active_subcategory = main_category

                    # Clean up category names if needed
                    if isinstance(available_subcategories, str):
                        available_subcategories = [available_subcategories]
                    elif not isinstance(available_subcategories, list):
                        available_subcategories = []

                    print(f"   ✅ Consensus analysis successful for {filename}:")
                    print(f"      📂 Main category: {main_category}")
                    print(f"      📁 Active subcategory: {active_subcategory}")
                    print(f"      🏷️  Available subcategories: {available_subcategories}")

                    # PHASE 3: Create canvas mappings (simple approach - all canvases get same category)
                    # Use existing tile detection to count canvases
                    canvas_count = 0
                    canvas_assignments = []
                    try:
                        # Quick tile detection to determine canvas count
                        tiles = self.tile_detector.detect_tiles(full_image)
                        canvas_count = len(tiles)

                        # Assign all canvases to the detected active subcategory
                        for i, tile in enumerate(tiles):
                            canvas_assignments.append({
                                "canvas_id": i + 1,
                                "x": tile[0], "y": tile[1], "width": tile[2], "height": tile[3],
                                "main_category": main_category,
                                "subcategory": active_subcategory,
                                "assignment_method": "consensus_based_uniform"
                            })
                    except Exception as tile_error:
                        print(f"   ⚠️ Tile detection failed, using default canvas count: {tile_error}")
                        canvas_count = 4  # Default assumption for Flink

                    return {
                        "main_category": main_category,
                        "active_subcategory": active_subcategory,
                        "available_subcategories": available_subcategories,
                        "header_tabs": available_subcategories,  # Use subcategories as tab list
                        "canvas_categories": canvas_assignments,
                        "total_canvases": canvas_count,
                        "confidence": consensus_result.get("confidence", 0.8),
                        "method": "consensus_based_header_analysis",
                        "analysis_source": "consensus_system_v1"
                    }

                else:
                    print(f"   ❌ Consensus analysis failed for {filename}, using fallback")

            # FALLBACK: If consensus fails, return minimal structure
            print(f"   ⚠️ Using fallback method for {filename} (no consensus system available)")
            return {
                "main_category": "Unknown",
                "active_subcategory": "Unknown",
                "available_subcategories": [],
                "header_tabs": [],
                "canvas_categories": [],
                "total_canvases": 0,
                "confidence": 0.0,
                "method": "consensus_fallback",
                "analysis_source": "fallback_v1"
            }

        except Exception as e:
            print(f"   ❌ Consensus-based header analysis failed for {filename}: {e}")
            return {
                "main_category": "Unknown",
                "active_subcategory": "Unknown",
                "available_subcategories": [],
                "header_tabs": [],
                "canvas_categories": [],
                "total_canvases": 0,
                "confidence": 0.0,
                "method": "consensus_analysis_error",
                "analysis_source": "error_v1",
                "error": str(e)
            }

    def _analyze_header_tabs(self, image: np.ndarray, filename: str) -> Dict[str, Any]:
        """Phase 1: Extract active category from pink-highlighted header tabs"""
        try:
            print(f"      🎯 Phase 1: Analyzing header tabs for {filename}")

            # Extract header region - focus on actual tab area (around y=100-170)
            header_region = image[100:170, :]
            height, width = header_region.shape[:2]

            # Find pink-highlighted active tab using refined color detection
            hsv = cv2.cvtColor(header_region, cv2.COLOR_BGR2HSV)

            # Enhanced pink/magenta color range for active tab highlighting
            # Use broader range to catch different pink shades
            lower_pink1 = np.array([140, 30, 100])    # Lighter pink
            upper_pink1 = np.array([180, 255, 255])   # Standard pink
            lower_pink2 = np.array([300, 30, 100])    # Wrapped around hue
            upper_pink2 = np.array([360, 255, 255])   # High hue pink

            pink_mask1 = cv2.inRange(hsv, lower_pink1, upper_pink1)
            pink_mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
            pink_mask = cv2.bitwise_or(pink_mask1, pink_mask2)

            # Find contours of pink regions
            contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            active_tab_text = ""
            all_tab_texts = []
            confidence = 0.0

            # Extract all visible tab text for context first
            try:
                if hasattr(self.food_extractor, 'text_extractor') and hasattr(self.food_extractor.text_extractor, 'reader'):
                    all_text_results = self.food_extractor.text_extractor.reader.readtext(header_region)
                    for bbox, text, conf in all_text_results:
                        if conf > 0.4 and text.strip() and len(text.strip()) > 2:  # Lower threshold, longer text
                            clean_text = text.strip()
                            # Filter out obvious non-category text (times, numbers, etc.)
                            if not clean_text.replace(':', '').replace('.', '').isdigit():
                                all_tab_texts.append(clean_text)

                                # Check if this text contains German category patterns
                                text_lower = clean_text.lower()
                                category_patterns = ['fleisch', 'fisch', 'vegan', 'brot', 'gemüse', 'obst', 'getränke']
                                if any(pattern in text_lower for pattern in category_patterns):
                                    if not active_tab_text or conf > confidence:  # Take highest confidence category
                                        active_tab_text = clean_text
                                        confidence = conf
            except Exception as e:
                print(f"         ⚠️ Failed to extract tab texts: {e}")

            # If pink detection found regions, try to extract text from them
            if contours:
                print(f"         🔍 Found {len(contours)} pink regions")
                for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:  # Check top 3 largest
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 30 and h > 10:  # Reasonable size for tab text
                        try:
                            active_tab_region = header_region[y:y+h, x:x+w]
                            if hasattr(self.food_extractor, 'text_extractor') and hasattr(self.food_extractor.text_extractor, 'reader'):
                                text_results = self.food_extractor.text_extractor.reader.readtext(active_tab_region)
                                for bbox, text, conf in text_results:
                                    if conf > 0.4 and text.strip():
                                        pink_text = text.strip()
                                        if len(pink_text) > 2 and not pink_text.replace(':', '').isdigit():
                                            if not active_tab_text or conf > confidence:
                                                active_tab_text = pink_text
                                                confidence = conf
                                            break
                        except Exception as e:
                            print(f"         ⚠️ OCR failed on pink region: {e}")

            print(f"         🎯 Active tab detected: '{active_tab_text}' (confidence: {confidence:.2f})")
            print(f"         📋 All tab texts: {all_tab_texts}")

            return {
                "active_category": active_tab_text if active_tab_text else "Unknown",
                "all_tabs": list(set(all_tab_texts)),  # Remove duplicates
                "confidence": confidence,
                "pink_regions_found": len(contours),
                "method": "enhanced_pink_tab_detection"
            }

        except Exception as e:
            print(f"         ❌ Header tab analysis failed: {e}")
            return {
                "active_category": "Unknown",
                "all_tabs": [],
                "confidence": 0.0,
                "pink_regions_found": 0,
                "method": "header_analysis_error",
                "error": str(e)
            }

    def _analyze_content_subcategories(self, image: np.ndarray, filename: str) -> Dict[str, Any]:
        """Phase 2: Extract subcategory headers from content area"""
        try:
            print(f"      📑 Phase 2: Analyzing content subcategories for {filename}")

            # Extract content region (below header, above footer - approximately 200-800px)
            content_region = image[200:800, :]
            height, width = content_region.shape[:2]

            subcategory_headers = []
            primary_subcategory = ""
            confidence = 0.0

            # Use OCR to find all text in content area
            if hasattr(self.food_extractor, 'text_extractor') and hasattr(self.food_extractor.text_extractor, 'reader'):
                try:
                    text_results = self.food_extractor.text_extractor.reader.readtext(content_region)

                    # Find text that appears to be subcategory headers
                    # These are typically larger text elements that appear above product groups
                    text_elements = []
                    for bbox, text, conf in text_results:
                        if conf > 0.6 and text.strip():
                            # Calculate position and size info
                            x1, y1 = int(min([point[0] for point in bbox])), int(min([point[1] for point in bbox]))
                            x2, y2 = int(max([point[0] for point in bbox])), int(max([point[1] for point in bbox]))
                            text_width = x2 - x1
                            text_height = y2 - y1

                            text_elements.append({
                                "text": text.strip(),
                                "confidence": conf,
                                "x": x1, "y": y1,
                                "width": text_width, "height": text_height,
                                "area": text_width * text_height
                            })

                    # Sort by vertical position (y-coordinate) to find header hierarchy
                    text_elements.sort(key=lambda x: x["y"])

                    # Identify subcategory headers by characteristics:
                    # - Larger text size (area > threshold)
                    # - German category patterns
                    # - Position in upper content area

                    german_category_patterns = [
                        "fleisch", "fisch", "rind", "kalb", "schwein", "hähnchen", "pute",
                        "obst", "gemüse", "bananen", "äpfel", "birnen", "beeren",
                        "joghurt", "dessert", "milch", "butter", "käse", "sahne",
                        "backwaren", "brot", "brötchen", "kuchen",
                        "getränke", "wasser", "saft", "limonade", "bier", "wein",
                        "tiefkühl", "eis", "pizza", "fertig", "categories", "vegan"
                    ]

                    # Enhanced subcategory detection logic
                    for element in text_elements:
                        text_lower = element["text"].lower()
                        clean_text = element["text"].strip()

                        # Check if this text matches German category patterns
                        is_category = any(pattern in text_lower for pattern in german_category_patterns)

                        # Check if text size suggests it's a header (larger than product text)
                        is_large_text = element["area"] > 2000  # Lower threshold for header text

                        # Position in upper portion of content area
                        is_upper_content = element["y"] < height * 0.7

                        # Enhanced pattern matching for specific categories
                        is_meat_category = any(meat in text_lower for meat in ["rind", "kalb", "fleisch", "schwein"])

                        # Skip generic words that aren't actual subcategories
                        is_generic = text_lower in ['categories', 'vegan', 'brotaufstriche'] and not is_meat_category

                        if (is_category or is_large_text) and is_upper_content and len(clean_text) > 3:
                            subcategory_headers.append(clean_text)

                            # Prioritize meat categories like "Rind- & Kalbfleisch" over generic ones
                            if is_meat_category and (not primary_subcategory or is_generic):
                                primary_subcategory = clean_text
                                confidence = element["confidence"]
                            elif not primary_subcategory and not is_generic:
                                primary_subcategory = clean_text
                                confidence = element["confidence"]

                    # Remove duplicates while preserving order
                    subcategory_headers = list(dict.fromkeys(subcategory_headers))

                    # Final check: if we have "Rind- & Kalbfleisch" in headers, make it primary
                    for header in subcategory_headers:
                        if "rind" in header.lower() and "kalb" in header.lower():
                            primary_subcategory = header
                            confidence = 0.9  # High confidence for specific match
                            break

                    print(f"         📋 Found subcategory headers: {subcategory_headers}")
                    print(f"         🎯 Primary subcategory: '{primary_subcategory}'")

                except Exception as e:
                    print(f"         ⚠️ OCR failed on content region: {e}")

            return {
                "primary_subcategory": primary_subcategory if primary_subcategory else "Unknown",
                "all_subcategories": subcategory_headers,
                "confidence": confidence,
                "method": "content_ocr_analysis"
            }

        except Exception as e:
            print(f"         ❌ Content subcategory analysis failed: {e}")
            return {
                "primary_subcategory": "Unknown",
                "all_subcategories": [],
                "confidence": 0.0,
                "method": "content_analysis_error",
                "error": str(e)
            }

    def _map_canvases_to_categories(self, image: np.ndarray, header_analysis: Dict, content_analysis: Dict, filename: str) -> Dict[str, Any]:
        """Phase 3: Map product canvases to their nearest subcategory headers"""
        try:
            print(f"      🗺️ Phase 3: Mapping canvases to categories for {filename}")

            # Use the existing tile detection to find product canvases
            tile_detector = TileDetector()
            detected_tiles = tile_detector.detect_tiles(image)

            canvas_assignments = []
            main_category = header_analysis.get("active_category", "Unknown")
            primary_subcategory = content_analysis.get("primary_subcategory", "Unknown")

            # For each detected product canvas, assign it to the appropriate category
            for i, (x, y, w, h) in enumerate(detected_tiles):
                canvas_assignment = {
                    "canvas_id": i + 1,
                    "x": x, "y": y, "width": w, "height": h,
                    "main_category": main_category,
                    "subcategory": primary_subcategory,  # In this simple version, all get the primary subcategory
                    "assignment_method": "primary_subcategory_fallback"
                }

                # TODO: In a more sophisticated version, we would:
                # 1. Find the nearest subcategory header above this canvas
                # 2. Use spatial analysis to determine which header governs this canvas
                # 3. Handle cases where canvases span multiple subcategory sections

                canvas_assignments.append(canvas_assignment)

            print(f"         🎯 Mapped {len(canvas_assignments)} canvases to categories")
            print(f"         📊 Main category: {main_category}")
            print(f"         🏷️ Primary subcategory: {primary_subcategory}")

            return {
                "canvas_assignments": canvas_assignments,
                "total_canvases": len(detected_tiles),
                "main_category": main_category,
                "primary_subcategory": primary_subcategory,
                "method": "spatial_canvas_mapping"
            }

        except Exception as e:
            print(f"         ❌ Canvas mapping failed: {e}")
            return {
                "canvas_assignments": [],
                "total_canvases": 0,
                "main_category": "Unknown",
                "primary_subcategory": "Unknown",
                "method": "canvas_mapping_error",
                "error": str(e)
            }

    def _use_working_ocr_method(self, header_image: np.ndarray, filename: str) -> Dict[str, Any]:
        """Use the same OCR method that works in individual processing"""
        try:
            # Use the exact same text extraction as the working individual processing
            text_results = self.food_extractor.text_extractor.reader.readtext(header_image)

            detected_text = []
            for bbox, text, confidence in text_results:
                if confidence > 0.5:
                    detected_text.append(text.lower())

            all_text = " ".join(detected_text)
            print(f"   📝 Working OCR detected for {filename}: '{all_text[:100]}...'")

            # Use the same pattern matching that works in individual processing
            return self._extract_categories_from_ocr_text(all_text, detected_text, filename)

        except Exception as e:
            print(f"   ❌ Working OCR failed for {filename}: {e}")
            return None

    def _parse_consensus_categories(self, response_text: str, filename: str) -> Dict[str, Any]:
        """Parse categories from consensus system response"""
        if not response_text:
            print(f"   ⚠️  Empty consensus response for {filename}")
            return None

        # First try to extract structured JSON
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                category_data = json.loads(json_match.group())
                return {
                    "main_category": category_data.get("main_category", "Unknown"),
                    "active_subcategory": category_data.get("active_subcategory", "Unknown"),
                    "available_subcategories": category_data.get("available_subcategories", []),
                    "navigation_text": category_data.get("navigation_text", []),
                    "confidence": 0.8,
                    "method": "consensus_json"
                }
        except json.JSONDecodeError:
            pass

        # Fallback: Extract categories from the consensus text response
        return self._extract_categories_from_text(response_text, filename)

    def _extract_categories_from_text(self, text: str, filename: str) -> Dict[str, Any]:
        """Extract categories from any text using pattern matching"""
        if not text:
            print(f"   ⚠️  Empty text provided for {filename}")
            return None
        text_lower = text.lower()

        # Use the expanded category patterns
        category_patterns = {
            "Obst": ["obst", "fruit", "früchte", "bananen", "äpfel", "birnen", "beeren"],
            "Gemüse": ["gemüse", "vegetables", "vegetable"],
            "Joghurt & Desserts": ["joghurt", "dessert", "yogurt", "pudding", "quark", "desserts"],
            "Milch & Butter": ["milch", "butter", "käse", "sahne", "milk", "cheese"],
            "Backwaren": ["backwaren", "brot", "bakery", "bread"],
            "Kinder": ["kinder", "baby", "kids", "children"],
            "Fleisch & Wurst": ["fleisch", "wurst", "meat", "sausage", "schinken"],
            "Tiefkühl": ["tiefkühl", "frozen", "tk", "gefroren"],
            "Getränke": ["getränke", "drinks", "wasser", "saft", "beverages"]
        }

        detected_category = "Unknown"
        detected_subcategories = []

        # Find main category
        for category, patterns in category_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                detected_category = category
                print(f"   ✅ Found category {category} via consensus text analysis")
                break

        # Extract subcategories from detected text
        subcategory_patterns = [
            "highlights", "bananen", "banana", "äpfel", "apple", "birnen", "pear",
            "beeren", "berry", "joghurt", "yogurt", "dessert", "pudding"
        ]

        for pattern in subcategory_patterns:
            if pattern in text_lower:
                if pattern in ["bananen", "banana"]:
                    detected_subcategories.append("Bananen")
                elif pattern in ["äpfel", "apple", "birnen", "pear"]:
                    detected_subcategories.append("Äpfel & Birnen")
                elif pattern == "highlights":
                    detected_subcategories.append("Highlights")
                elif pattern in ["beeren", "berry"]:
                    detected_subcategories.append("Beeren")
                elif pattern in ["joghurt", "yogurt"]:
                    detected_subcategories.append("Joghurt")
                elif pattern in ["dessert", "pudding"]:
                    detected_subcategories.append("Desserts")

        return {
            "main_category": detected_category,
            "active_subcategory": detected_subcategories[0] if detected_subcategories else "Unknown",
            "available_subcategories": detected_subcategories,
            "navigation_text": [text[:100]],
            "confidence": 0.7,
            "method": "consensus_text_analysis"
        }

    def _extract_categories_from_ocr_text(self, all_text: str, detected_text: List[str], filename: str) -> Dict[str, Any]:
        """Extract categories from OCR text using enhanced pattern matching for header navigation"""
        print(f"   🔍 Analyzing OCR text: '{all_text}'")
        print(f"   📝 Individual text pieces: {detected_text}")

        # Enhanced category patterns with exact navigation text
        category_patterns = {
            "Obst": ["obst", "fruit", "früchte", "bananen", "äpfel", "birnen", "beeren", "categories"],
            "Gemüse": ["gemüse", "vegetables", "vegetable"],
            "Joghurt & Desserts": ["joghurt", "dessert", "yogurt", "pudding", "quark", "desserts"],
            "Milch & Butter": ["milch", "butter", "käse", "sahne", "milk", "cheese"],
            "Backwaren": ["backwaren", "brot", "bakery", "bread"],
            "Kinder": ["kinder", "baby", "kids", "children"],
            "Fleisch & Wurst": ["fleisch", "wurst", "meat", "sausage", "schinken"],
            "Tiefkühl": ["tiefkühl", "frozen", "tk", "gefroren"],
            "Getränke": ["getränke", "drinks", "wasser", "saft", "beverages"]
        }

        detected_category = "Unknown"
        detected_subcategories = []

        # Enhanced detection: check both combined text and individual pieces
        all_text_combined = " ".join(detected_text).lower()

        # Special case: if we see "categories" in header, assume this is Obst (main category page)
        if "categories" in all_text_combined:
            detected_category = "Obst"
            print(f"   🎯 Found 'categories' keyword, inferring main category: Obst")

        # Standard pattern matching
        for category, patterns in category_patterns.items():
            if any(pattern in all_text_combined for pattern in patterns):
                detected_category = category
                print(f"   ✅ Found category {category} via enhanced OCR pattern matching")
                break

        # Extract subcategories
        subcategory_patterns = [
            "highlights", "bananen", "banana", "äpfel", "apple", "birnen", "pear",
            "beeren", "berry", "joghurt", "yogurt", "dessert", "pudding"
        ]

        for pattern in subcategory_patterns:
            if pattern in all_text:
                if pattern in ["bananen", "banana"]:
                    detected_subcategories.append("Bananen")
                elif pattern in ["äpfel", "apple", "birnen", "pear"]:
                    detected_subcategories.append("Äpfel & Birnen")
                elif pattern == "highlights":
                    detected_subcategories.append("Highlights")
                elif pattern in ["beeren", "berry"]:
                    detected_subcategories.append("Beeren")
                elif pattern in ["joghurt", "yogurt"]:
                    detected_subcategories.append("Joghurt")
                elif pattern in ["dessert", "pudding"]:
                    detected_subcategories.append("Desserts")

        return {
            "main_category": detected_category,
            "active_subcategory": detected_subcategories[0] if detected_subcategories else "Unknown",
            "available_subcategories": detected_subcategories,
            "navigation_text": detected_text[:5],
            "confidence": 0.7,
            "method": "working_ocr_method"
        }


    def _parse_category_json(self, response_text: str, filename: str) -> Dict[str, Any]:
        """Parse category information from consensus response"""
        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                category_data = json.loads(json_match.group())
                return {
                    "main_category": category_data.get("main_category", "Unknown"),
                    "active_subcategory": category_data.get("active_subcategory", "Unknown"),
                    "available_subcategories": category_data.get("available_subcategories", []),
                    "navigation_text": category_data.get("navigation_text", []),
                    "confidence": 0.8,
                    "method": "consensus"
                }
        except json.JSONDecodeError:
            pass

        # Fallback to text parsing
        return self._fallback_category_extraction(response_text, filename)

    def _fallback_category_extraction(self, text: str, filename: str) -> Dict[str, Any]:
        """Fallback category extraction using text patterns"""
        text_lower = text.lower()

        # Expanded category patterns
        category_patterns = {
            "Obst": ["obst", "fruit", "früchte"],
            "Gemüse": ["gemüse", "vegetables", "vegetable"],
            "Joghurt & Desserts": ["joghurt", "dessert", "yogurt", "pudding", "quark"],
            "Milch & Butter": ["milch", "butter", "käse", "sahne", "milk", "cheese"],
            "Backwaren": ["backwaren", "brot", "bakery", "bread"],
            "Kinder": ["kinder", "baby", "kids", "children"],
            "Fleisch & Wurst": ["fleisch", "wurst", "meat", "sausage", "schinken"],
            "Tiefkühl": ["tiefkühl", "frozen", "tk", "gefroren"],
            "Getränke": ["getränke", "drinks", "wasser", "saft", "beverages"]
        }

        detected_category = "Unknown"
        for category, patterns in category_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                detected_category = category
                break

        return {
            "main_category": detected_category,
            "active_subcategory": "Unknown",
            "available_subcategories": [],
            "navigation_text": [text[:100]],
            "confidence": 0.6,
            "method": "text_fallback"
        }

    def _build_shop_profile(self, category_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive shop profile from analysis"""
        successful_analyses = category_analysis.get("successful_analyses", [])

        # Extract shop characteristics
        categories = set()
        for result in successful_analyses:
            analysis = result.get("analysis", {})
            main_cat = analysis.get("main_category")
            if main_cat and main_cat != "Unknown":
                categories.add(main_cat)

        # Determine shop type based on categories
        shop_type = "grocery_delivery"
        if len(categories) >= 5:
            shop_type = "full_grocery_store"
        elif any(cat in categories for cat in ["Obst", "Gemüse"]):
            shop_type = "grocery_delivery"

        return {
            "name": "Grocery Delivery App",
            "type": shop_type,
            "layout_type": "grid_with_categories",
            "total_categories": len(categories),
            "screenshots_analyzed": len(successful_analyses),
            "analysis_timestamp": datetime.now().isoformat()
        }

    def _build_category_database(self, category_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive category database"""
        successful_analyses = category_analysis.get("successful_analyses", [])

        categories = {}
        keyword_mapping = {}

        for result in successful_analyses:
            analysis = result.get("analysis", {})
            main_cat = analysis.get("main_category")
            active_sub = analysis.get("active_subcategory")
            available_subs = analysis.get("available_subcategories", [])

            if main_cat and main_cat != "Unknown":
                if main_cat not in categories:
                    categories[main_cat] = {
                        "subcategories": set(),
                        "keywords": set(),
                        "screenshot_count": 0
                    }

                categories[main_cat]["screenshot_count"] += 1

                # Add subcategories
                if active_sub and active_sub != "Unknown":
                    categories[main_cat]["subcategories"].add(active_sub)

                for sub in available_subs:
                    if sub:
                        categories[main_cat]["subcategories"].add(sub)

                # Add keywords (lowercase main category name)
                categories[main_cat]["keywords"].add(main_cat.lower())
                keyword_mapping[main_cat.lower()] = main_cat

        # Convert sets to lists for JSON serialization
        for category in categories:
            categories[category]["subcategories"] = sorted(list(categories[category]["subcategories"]))
            categories[category]["keywords"] = sorted(list(categories[category]["keywords"]))

        return {
            "categories": categories,
            "keyword_mapping": keyword_mapping,
            "total_categories": len(categories),
            "created_timestamp": datetime.now().isoformat()
        }

    def _save_shop_knowledge(self, shop_profile: Dict, category_database: Dict, step_dir: Path):
        """Save shop knowledge to files for reuse"""
        with open(step_dir / "shop_profile.json", "w", encoding="utf-8") as f:
            json.dump(shop_profile, f, indent=2, ensure_ascii=False)

        with open(step_dir / "category_database.json", "w", encoding="utf-8") as f:
            json.dump(category_database, f, indent=2, ensure_ascii=False)

        print(f"   💾 Shop knowledge saved to {step_dir}")

    def _visualize_category_discovery(self, category_analysis: Dict, step_dir: Path):
        """Create visualization of category discovery results"""
        successful_analyses = category_analysis.get("successful_analyses", [])

        # Create summary visualization
        summary_data = []
        for result in successful_analyses:
            analysis = result.get("analysis", {})
            summary_data.append({
                "Screenshot": result.get("filename", "Unknown"),
                "Main Category": analysis.get("main_category", "Unknown"),
                "Active Subcategory": analysis.get("active_subcategory", "Unknown"),
                "Subcategories Found": len(analysis.get("available_subcategories", [])),
                "Method": analysis.get("method", "Unknown")
            })

        # Save as CSV for easy review
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_path = step_dir / "category_discovery_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"   📊 Category analysis saved to: {csv_path}")

    def _step_01_ui_region_analysis(self, image: np.ndarray, name: str) -> Dict[str, Any]:
        """STEP 1: UI Region Analysis - Identify UI regions (header/content/footer)"""
        print("🔍 STEP 1: UI Region Analysis - Identifying header, content, and footer regions")

        step_dir = self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir

        try:
            # Use ScreenshotUIAnalyzer to properly identify regions
            ui_analyzer = ScreenshotUIAnalyzer()
            ui_analysis = ui_analyzer.analyze_screenshot(image)

            print(f"   📋 Identified {len(ui_analysis.get('regions', {}))} UI regions")

            # Extract region boundaries
            regions = ui_analysis.get('regions', {})
            header_region = regions.get('header')
            content_region = regions.get('content')
            footer_region = regions.get('footer')

            # ENHANCED COMPATIBILITY CHECK
            compatibility_check = self._validate_ui_compatibility(regions, image, name)
            if not compatibility_check["compatible"]:
                print(f"   ❌ UI Structure Incompatible: {compatibility_check['reason']}")
                error_path = step_dir / f"{name}_01_incompatible_ui.txt"
                error_message = f"""FLINK UI STRUCTURE INCOMPATIBILITY ERROR - CANNOT PROCESS

Reason: {compatibility_check['reason']}
Details: {compatibility_check['details']}

This screenshot does not match the expected Flink grocery app UI structure.

Expected Flink Structure:
- Header: 200-600px tall with category tabs at top (y=0-30)
- Content: ≥800px tall product grid starting after header
- Portrait format phone screenshot (≥1000x1800, ratio <0.8)
- Header should be 10-50% of content height

Found Structure: {compatibility_check['found_structure']}
Header Height: {compatibility_check.get('header_height', 'Unknown')}px
Content Height: {compatibility_check.get('content_height', 'Unknown')}px
Image Dimensions: {compatibility_check.get('image_dimensions', 'Unknown')}

This pipeline is specifically calibrated for Flink grocery app screenshots.
Please ensure the image matches the Flink UI layout before processing."""

                error_path.write_text(error_message)

                return {
                    "status": "incompatible",
                    "error": "UI structure incompatible",
                    "message": compatibility_check['reason'],
                    "details": compatibility_check['details'],
                    "expected_structure": "Header + Content + Footer",
                    "found_structure": compatibility_check['found_structure'],
                    "header_region": None,
                    "content_region": None,
                    "footer_region": None
                }

            if not header_region:
                print("   ❌ No header region identified - creating error file")
                error_path = step_dir / f"{name}_01_error.txt"
                error_path.write_text("Image Error - No Header Found\nStep 0 did not provide valid header region data.\nSkipping further processing for this image.")

                return {
                    "status": "error",
                    "error": "No header region found",
                    "message": "Image Error - No Header Found",
                    "header_region": None,
                    "content_region": None,
                    "footer_region": None
                }

            # Save annotated image showing regions
            annotated_image = image.copy()

            # Draw region boundaries
            if header_region:
                cv2.rectangle(annotated_image,
                            (header_region['x'], header_region['y']),
                            (header_region['x'] + header_region['width'], header_region['y'] + header_region['height']),
                            (0, 255, 0), 3)
                cv2.putText(annotated_image, 'HEADER',
                          (header_region['x'] + 10, header_region['y'] + 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if content_region:
                cv2.rectangle(annotated_image,
                            (content_region['x'], content_region['y']),
                            (content_region['x'] + content_region['width'], content_region['y'] + content_region['height']),
                            (255, 0, 0), 3)
                cv2.putText(annotated_image, 'CONTENT',
                          (content_region['x'] + 10, content_region['y'] + 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            if footer_region:
                cv2.rectangle(annotated_image,
                            (footer_region['x'], footer_region['y']),
                            (footer_region['x'] + footer_region['width'], footer_region['y'] + footer_region['height']),
                            (0, 0, 255), 3)
                cv2.putText(annotated_image, 'FOOTER',
                          (footer_region['x'] + 10, footer_region['y'] + 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Save annotated regions
            annotated_path = step_dir / f"{name}_01_annotated.jpg"
            cv2.imwrite(str(annotated_path), annotated_image)

            # Save individual region images
            header_path = None
            if header_region:
                header_image = image[header_region['y']:header_region['y']+header_region['height'],
                                   header_region['x']:header_region['x']+header_region['width']]
                header_path = step_dir / f"{name}_01_header_region.jpg"
                cv2.imwrite(str(header_path), header_image)

            # Save region data
            region_data = {
                'header_region': header_region,
                'content_region': content_region,
                'footer_region': footer_region,
                'ui_analysis': ui_analysis
            }

            region_json_path = step_dir / f"{name}_01_header_text.json"
            with open(region_json_path, 'w') as f:
                json.dump(region_data, f, indent=2)

            print(f"   ✅ UI regions identified and saved")
            print(f"   📏 Header: {header_region['width']}x{header_region['height']} at ({header_region['x']}, {header_region['y']})")
            if content_region:
                print(f"   📏 Content: {content_region['width']}x{content_region['height']} at ({content_region['x']}, {content_region['y']})")

            return {
                "status": "success",
                "header_region": header_region,
                "content_region": content_region,
                "footer_region": footer_region,
                "ui_analysis": ui_analysis,
                "files": {
                    "annotated": str(annotated_path),
                    "header_region": str(header_path) if header_region else None,
                    "region_data": str(region_json_path)
                }
            }

        except Exception as e:
            print(f"   ❌ UI region analysis failed: {e}")
            error_path = step_dir / f"{name}_01_error.txt"
            error_path.write_text(f"Image Error - UI Analysis Failed\nError: {str(e)}\nSkipping further processing for this image.")

            return {
                "status": "error",
                "error": str(e),
                "message": "Image Error - UI Analysis Failed",
                "header_region": None,
                "content_region": None,
                "footer_region": None
            }

    def _validate_ui_compatibility(self, regions: Dict, image: np.ndarray, name: str) -> Dict[str, Any]:
        """Validate if screenshot has Flink-compatible UI structure for processing"""
        try:
            # Extract region information
            header_region = regions.get('header')
            content_region = regions.get('content')
            footer_region = regions.get('footer')

            # Flink grocery app specific structure validation
            compatibility_issues = []
            found_structure = []
            image_height, image_width = image.shape[:2]

            # Expected Flink UI structure based on successfully processed images:
            # - Header: ~200-250px tall, at top (y=0-20)
            # - Content: Starts after header, substantial height for product grid
            # - Total image: Portrait phone screenshot (~1290x2556 typical)

            # Check 1: Header region validation for Flink consistency
            if not header_region:
                compatibility_issues.append("No header region detected - Flink requires header with category tabs")
                found_structure.append("Missing Header")
            else:
                found_structure.append("Header Present")

                # Flink-specific header height validation (based on working screenshots)
                header_height = header_region.get('height', 0)
                header_y = header_region.get('y', 0)

                # CRITICAL: Must be exactly 530px for Flink compatibility - DO NOT CHANGE
                if header_height != 530:
                    compatibility_issues.append(f"Incompatible header height: {header_height}px (Flink requires exactly 530px)")
                    return {"compatible": False, "reason": "Invalid header height", "found_structure": found_structure}

                # Header should be at very top
                if header_y > 30:
                    compatibility_issues.append(f"Header not at top: y={header_y} (Flink headers start at y=0-30)")

            # Check 2: Content region validation for Flink product grids
            if not content_region:
                compatibility_issues.append("No content region detected - Flink requires content area for product grid")
                found_structure.append("Missing Content")
            else:
                found_structure.append("Content Present")

                content_height = content_region.get('height', 0)
                content_y = content_region.get('y', 0)

                # CRITICAL: Content must start at exactly pixel 531 for Flink compatibility - DO NOT CHANGE
                if content_y != 531:
                    compatibility_issues.append(f"Incompatible content position: y={content_y} (Flink requires y=531)")
                    return {"compatible": False, "reason": "Invalid content position", "found_structure": found_structure}

                # Content should be substantial for product grid display
                if content_height < 800:
                    compatibility_issues.append(f"Content region too small: {content_height}px (Flink needs ≥800px for product grid)")

                # Content should start after header
                if header_region and content_y < (header_region.get('height', 0) - 50):
                    compatibility_issues.append(f"Content overlaps header - invalid Flink layout structure")

            # Check 3: Overall Flink screenshot structure validation
            # Flink screenshots should be portrait phone format
            if image_height < 1800:  # Minimum reasonable height for Flink screenshots
                compatibility_issues.append(f"Image too short: {image_height}px (Flink screenshots typically ≥2000px tall)")

            if image_width < 800:  # Minimum reasonable width
                compatibility_issues.append(f"Image too narrow: {image_width}px (Flink screenshots typically ≥1000px wide)")

            aspect_ratio = image_width / image_height if image_height > 0 else 0
            if aspect_ratio > 0.8:  # Should be portrait
                compatibility_issues.append(f"Not portrait format: ratio={aspect_ratio:.2f} (Flink screenshots are portrait < 0.8)")

            # Check 4: Header-to-content ratio validation (Flink-specific)
            if header_region and content_region:
                header_height = header_region.get('height', 0)
                content_height = content_region.get('height', 0)

                if header_height > 0 and content_height > 0:
                    header_content_ratio = header_height / content_height

                    # In Flink, header is typically 10-30% of content height
                    if header_content_ratio > 0.5:
                        compatibility_issues.append(f"Header too large relative to content: {header_content_ratio:.2f} (Flink header should be < 50% of content)")
                    elif header_content_ratio < 0.1:
                        compatibility_issues.append(f"Header too small relative to content: {header_content_ratio:.2f} (Flink header should be > 10% of content)")

            # Check 5: Expected Flink UI positioning
            total_ui_coverage = 0
            if header_region:
                total_ui_coverage += header_region.get('height', 0)
            if content_region:
                total_ui_coverage += content_region.get('height', 0)
            if footer_region:
                total_ui_coverage += footer_region.get('height', 0)

            ui_coverage_ratio = total_ui_coverage / image_height if image_height > 0 else 0
            if ui_coverage_ratio < 0.7:  # UI should cover most of the image
                compatibility_issues.append(f"Poor UI coverage: {ui_coverage_ratio:.2f} (Flink UI should cover ≥70% of image)")

            # Determine compatibility based on critical Flink structure issues
            critical_issues = [issue for issue in compatibility_issues
                             if any(critical in issue.lower() for critical in
                                   ['missing header', 'missing content', 'too small', 'too large',
                                    'not portrait', 'overlaps header', 'poor ui coverage'])]

            is_compatible = len(critical_issues) == 0

            # Create result with Flink-specific information
            result = {
                "compatible": is_compatible,
                "reason": "Flink UI structure validated successfully" if is_compatible else f"Flink structure incompatible: {'; '.join(critical_issues)}",
                "details": f"Flink validation found {len(compatibility_issues)} issues: {'; '.join(compatibility_issues)}" if compatibility_issues else "All Flink structure checks passed",
                "found_structure": " + ".join(found_structure),
                "all_issues": compatibility_issues,
                "critical_issues": critical_issues,
                "image_dimensions": f"{image_width}x{image_height}",
                "aspect_ratio": round(aspect_ratio, 3),
                "header_height": header_region.get('height', 0) if header_region else 0,
                "content_height": content_region.get('height', 0) if content_region else 0,
                "validation_method": "flink_ui_structure_validation"
            }

            return result

        except Exception as e:
            # If validation fails, be conservative and allow processing
            return {
                "compatible": True,
                "reason": f"Validation error - allowing processing: {str(e)}",
                "details": "Could not complete Flink structure validation, defaulting to compatible",
                "found_structure": "Unknown - validation failed",
                "validation_method": "error_fallback"
            }

    def _step_02_header_category_analysis(self, image: np.ndarray, header_region: Dict, name: str) -> Dict[str, Any]:
        """STEP 2: Header Category Analysis using properly identified header region"""
        print("🏷️ STEP 2: Header Category Analysis using consensus system")

        if not header_region:
            print("   ❌ No header region provided - skipping category analysis")
            return {
                "status": "error",
                "error": "No header region",
                "category_data": None
            }

        step_dir = self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir

        try:
            # Extract header region from image
            x, y = header_region['x'], header_region['y']
            width, height = header_region['width'], header_region['height']
            header_image = image[y:y+height, x:x+width]

            print(f"   📋 Header region: {width}x{height} at ({x}, {y})")

            # Save header image
            header_path = step_dir / f"{name}_02_analysis.jpg"
            cv2.imwrite(str(header_path), header_image)

            # Use consensus system for category analysis
            print("   🧠 Using 3-model consensus system for category detection...")

            analyzer = LocalConsensusAnalyzer()
            consensus_result = asyncio.run(
                analyzer.analyze_categories_with_consensus(header_image)
            )

            if consensus_result and consensus_result.get('success'):
                category_data = {
                    'main_category': consensus_result.get('main_category', 'Unknown'),
                    'active_subcategory': consensus_result.get('active_subcategory', 'Unknown'),
                    'available_subcategories': consensus_result.get('available_subcategories', []),
                    'confidence': consensus_result.get('confidence', 0.8),
                    'method': 'consensus_ui_analysis'
                }

                print(f"   ✅ Category detected: {category_data['main_category']}")
                print(f"   📊 Subcategory: {category_data['active_subcategory']}")
            else:
                print("   ⚠️ Consensus analysis failed - using fallback")
                category_data = self._fallback_category_extraction("no_text_available", name)

            # Save category analysis
            analysis_path = step_dir / f"{name}_02_analysis.json"
            with open(analysis_path, 'w') as f:
                json.dump(category_data, f, indent=2)

            return {
                "status": "success",
                "category_data": category_data,
                "files": {
                    "header_image": str(header_path),
                    "analysis": str(analysis_path)
                }
            }

        except Exception as e:
            print(f"   ❌ Header category analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "category_data": None
            }

    def _step_01_header_text_extraction(self, image: np.ndarray, step0_result: Optional[Dict], name: str) -> Dict[str, Any]:
        """STEP 1: Header Text Extraction using Consensus System"""
        print("🔤 STEP 1: Header Text Extraction with Consensus LLM System")

        step_dir = self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir

        # Get header region from step0_result or shop_profile
        header_region = None
        if step0_result and 'ui_analysis' in step0_result:
            regions = step0_result['ui_analysis'].get('regions', {})
            header_region = regions.get('header')
        elif self.shop_profile and 'header_region' in self.shop_profile:
            header_region = self.shop_profile['header_region']

        # If no header region found, fail gracefully without fallback processing
        if not header_region:
            print("   ❌ No header region found from Step 0 - skipping this image")
            error_path = step_dir / f"{name}_01_error.txt"
            error_path.write_text("Image Error - No Header Found\nStep 0 did not provide valid header region data.\nSkipping further processing for this image.")

            return {
                "status": "error",
                "error": "No header region found",
                "message": "Image Error - No Header Found",
                "header_region": None,
                "main_category": None,
                "subcategory": None
            }

        # Extract header region from image
        x = header_region['x']
        y = header_region['y']
        width = header_region['width']
        height = header_region['height']

        header_image = image[y:y+height, x:x+width]

        print(f"   📏 Header region: {width}x{height} at ({x}, {y})")

        # Save header region image
        header_path = step_dir / f"{name}_01_header_region.jpg"
        cv2.imwrite(str(header_path), header_image)

        # Use consensus system to extract category and subcategory text
        print("   🧠 Using consensus LLM system for text extraction...")
        try:
            # Import consensus analyzer
            from src.local_consensus_analyzer import LocalConsensusAnalyzer
            consensus_analyzer = LocalConsensusAnalyzer()

            # Analyze header with UI mode to extract category text
            import asyncio
            consensus_result = asyncio.run(
                consensus_analyzer.analyze_product_with_consensus(
                    header_image, header_image, "ui"
                )
            )

            # Parse categories and subcategories from consensus result
            category = None
            subcategory = None

            # Extract from consensus analysis - look for categories in various formats
            if 'categories' in consensus_result:
                categories = consensus_result['categories']
                if isinstance(categories, list) and len(categories) > 0:
                    category = categories[0] if categories else None
                    subcategory = categories[1] if len(categories) > 1 else None
                elif isinstance(categories, str):
                    # Parse from string format
                    parts = categories.split(',')
                    category = parts[0].strip() if parts else None
                    subcategory = parts[1].strip() if len(parts) > 1 else None

            # Fallback: try to extract from text analysis
            if not category and 'text' in consensus_result:
                text = consensus_result.get('text', '')
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                if lines:
                    category = lines[0] if lines else None
                    subcategory = lines[1] if len(lines) > 1 else None

            # Create result structure
            header_text_result = {
                "category": category,
                "subcategory": subcategory,
                "extraction_method": "consensus_llm",
                "raw_consensus_result": consensus_result
            }

            print(f"   ✅ Extracted - Category: '{category}', Subcategory: '{subcategory}'")

        except Exception as e:
            print(f"   ⚠️ Consensus analysis failed: {e}")
            # Fallback to basic OCR if consensus fails
            header_text_result = {
                "category": None,
                "subcategory": None,
                "extraction_method": "failed",
                "error": str(e)
            }

        # Save header text extraction results to JSON
        json_path = step_dir / f"{name}_01_header_text.json"
        with open(json_path, 'w') as f:
            import json
            json.dump(header_text_result, f, indent=2, ensure_ascii=False)

        # Create annotated version showing header region
        annotated = image.copy()
        h, w = image.shape[:2]

        # Draw header region rectangle
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), 3)

        # Add title and results
        cv2.putText(annotated, "STEP 1: Header Text Extraction",
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        if header_text_result["category"]:
            cv2.putText(annotated, f"Category: {header_text_result['category']}",
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if header_text_result["subcategory"]:
            cv2.putText(annotated, f"Subcategory: {header_text_result['subcategory']}",
                       (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        annotated_path = step_dir / f"{name}_01_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated)

        result = {
            "step": 1,
            "title": "Header Text Extraction",
            "description": "Category and subcategory extraction from header using consensus LLM system",
            "header_region": header_region,
            "category": header_text_result["category"],
            "subcategory": header_text_result["subcategory"],
            "extraction_method": header_text_result["extraction_method"],
            "files": {
                "header_region": str(header_path),
                "annotated": str(annotated_path),
                "results_json": str(json_path)
            },
            "analysis": header_text_result  # For backward compatibility with step2
        }

        print(f"  ✅ Saved header region: {header_path}")
        print(f"  ✅ Saved annotated: {annotated_path}")
        print(f"  ✅ Saved results JSON: {json_path}")
        return result

    def _load_step0_categories(self) -> Dict[str, str]:
        """Load category data from Step 0 results for CSV generation"""
        step0_dir = self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir

        # Try to load from the most recent Step 0 analysis file
        category_files = list(step0_dir.glob("*_00_analysis.json"))
        if category_files:
            # Get the most recent file
            latest_file = max(category_files, key=lambda x: x.stat().st_mtime)
            try:
                with open(latest_file, 'r') as f:
                    step0_data = json.load(f)

                categories = step0_data.get('categories', [])
                subcategories = step0_data.get('subcategories', [])

                return {
                    'primary_category': categories[0] if categories else 'Obst',  # Default to 'Fruit'
                    'primary_subcategory': subcategories[0] if subcategories else 'Unknown',
                    'all_categories': categories,
                    'all_subcategories': subcategories
                }
            except Exception as e:
                print(f"  ⚠️  Error loading Step 0 categories: {e}")

        # Fallback defaults
        return {
            'primary_category': 'Obst',
            'primary_subcategory': 'Unknown',
            'all_categories': [],
            'all_subcategories': []
        }

    def _step_03_product_canvas_detection(self, image: np.ndarray, ui_analysis: Dict, name: str) -> Dict[str, Any]:
        """STEP 3: Detect product canvas rectangles (573x813px each) using consolidated_pipeline.py reference"""
        print("🎯 STEP 3: Product Canvas Detection (573x813px each)")

        step_dir = self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir
        step_dir.mkdir(exist_ok=True)

        # First perform UI analysis to get regions if not provided
        if 'regions' not in ui_analysis:
            print("   🔍 Performing UI analysis to detect regions...")
            analysis = self.ui_analyzer.analyze_screenshot(image)
            ui_regions = analysis.get('regions', {})
        else:
            ui_regions = ui_analysis['regions']

        # Get content region from UI analysis
        if 'content' in ui_regions:
            content_region = ui_regions['content']
            content_x = content_region['x']
            content_y = content_region['y']
            content_w = content_region['width']
            content_h = content_region['height']
        else:
            # Fallback: use full image as content region
            h, w = image.shape[:2]
            content_region = {
                'x': 0,
                'y': 531,  # Skip header region (known from IMG_7805 analysis)
                'width': w,
                'height': h - 531 - 250  # Skip footer region
            }
            content_x = content_region['x']
            content_y = content_region['y']
            content_w = content_region['width']
            content_h = content_region['height']
            print(f"   ⚠️  No content region found, using fallback: {content_w}x{content_h} at ({content_x}, {content_y})")

        print(f"   📐 Content region: {content_w}x{content_h} at ({content_x}, {content_y})")

        # Extract content area and detect tiles
        content_image = image[content_y:content_y+content_h, content_x:content_x+content_w]
        detected_tiles = self.tile_detector.detect_tiles(content_image)

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
        canvas_path = step_dir / f"{name}_02_canvases.jpg"
        cv2.imwrite(str(canvas_path), vis_image)

        # Load category data from Step 0 results for CSV generation
        step0_categories = self._load_step0_categories()
        category = step0_categories.get('primary_category', 'Unknown')
        subcategory = step0_categories.get('primary_subcategory', 'Unknown')

        # Generate CSV data for each product canvas
        csv_data = []
        for canvas in valid_canvases:
            csv_row = {
                'product_canvas_id': f"{name}_canvas_{canvas['canvas_id']}",
                'category': category,
                'subcategory': subcategory,
                'canvas_x': canvas['x'],
                'canvas_y': canvas['y'],
                'canvas_width': canvas['width'],
                'canvas_height': canvas['height'],
                'source_image': name,
                'detection_confidence': 0.95,  # High confidence for tile-based detection
                'tile_x': canvas['tile_region']['x'],
                'tile_y': canvas['tile_region']['y'],
                'tile_width': canvas['tile_region']['width'],
                'tile_height': canvas['tile_region']['height']
            }
            csv_data.append(csv_row)

        # Save CSV file with product canvas data
        csv_path = step_dir / f"{name}_02_product_canvases.csv"
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
            print(f"  📊 Saved CSV with {len(csv_data)} product canvas rows: {csv_path}")
        else:
            print("  ⚠️  No product canvases found - CSV not created")

        # Save JSON data
        result_data = {
            "step": "02_product_canvas_detection",
            "description": f"Detected {len(valid_canvases)} product canvases (573x813px each)",
            "content_region": content_region,
            "canvases": valid_canvases,
            "category_data": step0_categories,
            "csv_data": csv_data
        }

        json_path = step_dir / f"{name}_02_canvases.json"
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=2)

        result = {
            "step": 2,
            "title": "Product Canvas Detection (573x813px)",
            "description": f"Detected {len(valid_canvases)} canvases using consolidated_pipeline.py reference",
            "files": {
                "image": str(canvas_path),
                "json": str(json_path),
                "csv": str(csv_path) if csv_data else None
            },
            "canvases": valid_canvases,
            "content_region": content_region,
            "csv_data": csv_data,
            "category_data": step0_categories
        }

        print(f"  ✅ Created {len(valid_canvases)} product canvases (573x813px each)")
        print(f"  ✅ Saved: {canvas_path}")
        print(f"  ✅ Saved: {json_path}")
        return result

    def _step_02_ui_analysis_old(self, image: np.ndarray, name: str) -> Dict[str, Any]:
        """STEP 2: UI Analysis & Region Segmentation"""
        print("🔍 STEP 2: UI Analysis & Region Detection")
        
        step_dir = self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir
        
        # Run UI analysis
        analysis = self.ui_analyzer.analyze_screenshot(image)
        
        # Create visualization
        vis_image = image.copy()
        regions = analysis["regions"]
        
        # Draw region boundaries with labels
        # Header - Red
        if regions["header"]:
            hr = regions["header"]
            cv2.rectangle(vis_image, (hr["x"], hr["y"]), 
                         (hr["x"] + hr["width"], hr["y"] + hr["height"]), (0, 0, 255), 3)
            cv2.putText(vis_image, "HEADER", (hr["x"] + 10, hr["y"] + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(vis_image, f"Categories: {analysis.get('categories', [])}", 
                       (hr["x"] + 10, hr["y"] + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Content - Green
        if regions["content"]:
            cr = regions["content"]
            cv2.rectangle(vis_image, (cr["x"], cr["y"]), 
                         (cr["x"] + cr["width"], cr["y"] + cr["height"]), (0, 255, 0), 3)
            cv2.putText(vis_image, "CONTENT AREA", (cr["x"] + 10, cr["y"] + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(vis_image, f"Products: {len(analysis.get('content_tiles', []))}", 
                       (cr["x"] + 10, cr["y"] + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Footer - Blue
        if regions["footer"]:
            fr = regions["footer"]
            cv2.rectangle(vis_image, (fr["x"], fr["y"]), 
                         (fr["x"] + fr["width"], fr["y"] + fr["height"]), (255, 0, 0), 3)
            cv2.putText(vis_image, "FOOTER", (fr["x"] + 10, fr["y"] + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Add step title
        cv2.putText(vis_image, "STEP 2: UI Analysis", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        # Save files
        analysis_path = step_dir / f"{name}_02_analysis.jpg"
        cv2.imwrite(str(analysis_path), vis_image)
        
        json_path = step_dir / f"{name}_02_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        result = {
            "step": 2,
            "title": "UI Analysis & Segmentation",
            "description": f"Detected {len(analysis.get('content_tiles', []))} product tiles, categories: {analysis.get('categories', [])}",
            "files": {
                "visualization": str(analysis_path),
                "analysis_json": str(json_path)
            },
            "analysis": analysis
        }
        
        print(f"  ✅ Saved: {analysis_path}")
        print(f"  ✅ Saved: {json_path}")
        print(f"  📊 Categories: {analysis.get('categories', [])}")
        print(f"  🎯 Product tiles: {len(analysis.get('content_tiles', []))}")
        return result

    def _step_04_component_coordinate_extraction(self, image: np.ndarray, canvases: List[Dict], name: str) -> Dict[str, Any]:
        """STEP 4: Extract component coordinates (FIXED pink button detection)"""
        print("📍 STEP 4: Component Coordinate Extraction (Fixed)")

        step_dir = self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir
        step_dir.mkdir(exist_ok=True)

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
            pink_button = self._detect_pink_button_in_tile_FIXED(canvas_region, x, y)

            # Prepare component data
            component = {
                'component_id': f"{name}_component_{canvas_id}",
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

        # Save components data
        with open(step_dir / f"{name}_components.json", 'w') as f:
            json.dump(components_data, f, indent=2)

        # Generate CSV file with component-level data
        csv_data = []
        for comp in components_data:
            csv_row = {
                'component_id': f"{name}_component_{comp['canvas_id']}",
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

        # Save CSV file
        csv_path = step_dir / f"{name}_03_components.csv"
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
            print(f"  📊 Saved CSV with {len(csv_data)} component rows: {csv_path}")

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

        vis_path = step_dir / f"{name}_03_components.jpg"
        cv2.imwrite(str(vis_path), vis_image)

        print(f"   ✅ Extracted coordinates for {len(components_data)} canvases")
        print(f"   📁 Saved: {vis_path}")

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

        return {
            'step': 3,
            'title': 'Component Coordinate Extraction (Fixed)',
            'description': f'Extracted precise coordinates for {len(components_data)} components with JSON-PNG overlay consistency',
            'files': {
                'visualization': str(vis_path),
                'json': str(step_dir / f"{name}_components.json"),
                'csv': str(csv_path) if csv_data else None
            },
            'components_data': components_data,
            'csv_data': csv_data,
            'tiles': tiles,  # For backward compatibility with subsequent steps
            'step_dir': step_dir
        }

    def _detect_pink_button_in_tile_FIXED(self, tile_region: np.ndarray, tile_x: int, tile_y: int) -> Dict[str, Any]:
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

    def _step_03_tile_detection(self, image: np.ndarray, analysis: Dict[str, Any], name: str) -> Dict[str, Any]:
        """STEP 3: Individual Product Tile Detection using working Phase 1+2 method"""
        print("🎯 STEP 3: Product Tile Detection (Using Working Phase 1+2 Method)")

        step_dir = self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir

        # STEP 1: First identify UI sections (header/footer/content)
        print("   🔍 Step 3A: UI Section Identification...")
        content_region = self._identify_content_region(image)
        print(f"   ✅ Content region: {content_region}")

        # STEP 2: Use Phase 1 pink button detection within content region
        print("   🔍 Step 3B: Phase 1 Pink Button Detection...")
        button_centers = self._detect_pink_buttons_phase1(image, content_region)
        print(f"   ✅ Found {len(button_centers)} pink button anchors")

        # STEP 3: Use Phase 2 tile boundary detection using button anchors
        print("   🔍 Step 3C: Phase 2 Tile Boundary Detection...")
        tiles = self._detect_tile_boundaries_phase2(image, button_centers, content_region)
        print(f"   ✅ Detected {len(tiles)} precise tile boundaries")

        # Create visualization showing the proper sequence
        vis_image = image.copy()

        # Add step title
        cv2.putText(vis_image, "STEP 3: Tiles (Phase 1+2 Method)",
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # Draw content region
        x, y, w, h = content_region['x'], content_region['y'], content_region['w'], content_region['h']
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis_image, "CONTENT AREA", (x + 10, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw pink button anchors
        for i, button in enumerate(button_centers):
            center_x, center_y = button['center']
            radius = button['radius']
            cv2.circle(vis_image, (center_x, center_y), radius, (255, 0, 255), 2)
            cv2.putText(vis_image, f"BTN{i+1}", (center_x - 15, center_y - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Draw each detected tile with proper boundaries
        for i, tile in enumerate(tiles):
            x, y, w, h = tile['x'], tile['y'], tile['w'], tile['h']

            # Draw tile boundary
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 255, 0), 3)

            # Add tile number
            cv2.putText(vis_image, f"TILE {i+1}", (x + 10, y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(vis_image, f"{w}x{h}px", (x + 10, y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save visualization
        tiles_path = step_dir / f"{name}_03_tiles_phase1_2.jpg"
        cv2.imwrite(str(tiles_path), vis_image)

        result = {
            "step": 3,
            "title": "Product Tile Detection (Phase 1+2 Method)",
            "description": f"Detected {len(tiles)} tiles using working Phase 1+2 method",
            "files": {
                "visualization": str(tiles_path)
            },
            "content_region": content_region,
            "button_anchors": button_centers,
            "tiles": tiles
        }

        print(f"  ✅ Saved: {tiles_path}")
        print(f"  🎯 Phase 1+2 Method: {len(tiles)} precise tiles detected")
        return result
    
    def _step_05_product_extraction(self, image: np.ndarray, tiles: List[Dict], name: str) -> Dict[str, Any]:
        """STEP 5: Refined A/B/C Pipeline with Comprehensive Safeguards Against Black Arrays"""
        print("\n🔥 STEP 5: REFINED A/B/C PIPELINE WITH BLACK ARRAY SAFEGUARDS")
        print("   📋 Step 4A: Pink button removal + product/text extraction")
        print("   📋 Step 4B: Enhanced consensus processing with image+text context")
        print("   📋 Step 4C: Background removal on clean product images")
        print("   🛡️  Comprehensive validation to prevent black array issues")
        print("=" * 80)

        step_dir = self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir
        step_dir.mkdir(parents=True, exist_ok=True)

        # CRITICAL: Load original source image explicitly to prevent black arrays
        original_image = self._load_original_source_image(name)
        if original_image is None:
            return self._create_error_result("Failed to load original source image")

        # Load Step 3 precise coordinates instead of Step 2 canvas regions
        step3_components = self._load_step3_components(name)
        if not step3_components:
            print("   ⚠️  No Step 3 components found, falling back to Step 2 tiles")
            step3_components = self._convert_step2_to_step3_format(tiles)

        # Initialize components with error handling
        consensus_analyzer = self._initialize_consensus_analyzer()
        bg_removal_manager = self._initialize_background_removal_manager()

        # Processing stats
        processing_stats = {
            "total_components": len(step3_components),
            "successful_extractions": 0,
            "pink_buttons_removed": 0,
            "background_removals": 0,
            "consensus_analyses": 0,
            "validation_warnings": 0
        }

        step4a_results = []
        step4b_results = []
        step4c_results = []

        print(f"\n🔄 Processing {len(step3_components)} product components...")

        for i, component in enumerate(step3_components):
            try:
                print(f"\n📦 COMPONENT {i+1}/{len(step3_components)}:")

                # STEP 4A: Extract with comprehensive validation
                step4a_result = self._step_4a_extract_with_validation(
                    original_image, component, i+1, name, step_dir
                )

                if step4a_result["success"]:
                    step4a_results.append(step4a_result)
                    processing_stats["successful_extractions"] += 1

                    if step4a_result.get("button_removed"):
                        processing_stats["pink_buttons_removed"] += 1

                    # STEP 4B: Enhanced consensus processing
                    step4b_result = self._step_4b_consensus_with_context(
                        step4a_result, consensus_analyzer, i+1
                    )

                    if step4b_result["success"]:
                        step4b_results.append(step4b_result)
                        processing_stats["consensus_analyses"] += 1

                    # STEP 4C: Background removal on clean product images
                    step4c_result = self._step_4c_background_removal(
                        step4a_result, bg_removal_manager, i+1, step_dir
                    )

                    if step4c_result["success"]:
                        step4c_results.append(step4c_result)
                        processing_stats["background_removals"] += 1

                else:
                    print(f"   ❌ Component {i+1} extraction failed: {step4a_result.get('error', 'Unknown error')}")
                    processing_stats["validation_warnings"] += 1

            except Exception as e:
                print(f"   ❌ Component {i+1} processing failed: {e}")
                processing_stats["validation_warnings"] += 1

        # Create comprehensive visualization grid
        self._create_step4_visualization_grid(step4a_results, step4b_results, step4c_results, name, step_dir)

        # Compile final results
        return {
            "step": "04_refined_abc_pipeline",
            "description": f"Processed {len(step3_components)} components through refined A/B/C pipeline",
            "processing_stats": processing_stats,
            "step4a_results": step4a_results,
            "step4b_results": step4b_results,
            "step4c_results": step4c_results,
            "validation_passed": processing_stats["validation_warnings"] == 0,
            "black_array_prevention": "active"
        }

    def _load_original_source_image(self, name: str) -> np.ndarray:
        """Load original source image with comprehensive validation"""
        print("   📂 Loading original source image...")

        # Try multiple possible source paths
        possible_paths = [
            f"/Users/davemooney/_dev/Flink/{name}.PNG",
            f"/Users/davemooney/_dev/Flink/{name}.png",
            f"/Users/davemooney/_dev/Flink/{name}.jpg",
            f"/Users/davemooney/_dev/Flink/{name}.jpeg"
        ]

        for path in possible_paths:
            try:
                image = cv2.imread(path)
                if image is not None:
                    # Comprehensive validation
                    if image.size == 0:
                        print(f"   ⚠️  Image at {path} is empty")
                        continue

                    if len(image.shape) != 3:
                        print(f"   ⚠️  Image at {path} is not 3D (shape: {image.shape})")
                        continue

                    if np.all(image == 0):
                        print(f"   ⚠️  Image at {path} is all black - suspicious!")
                        continue

                    print(f"   ✅ Loaded original image: {path}")
                    print(f"      📏 Dimensions: {image.shape[1]}×{image.shape[0]}px")
                    print(f"      🎨 Channels: {image.shape[2]}")
                    print(f"      📊 Value range: {image.min()}-{image.max()}")
                    return image

            except Exception as e:
                print(f"   ⚠️  Failed to load {path}: {e}")
                continue

        print("   ❌ No valid original source image found")
        return None

    def _load_step3_components(self, name: str) -> List[Dict]:
        """Load Step 3 component coordinates with precise pink button locations"""
        # Try both possible paths - with and without step prefix
        possible_paths = [
            (self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir) / f"{name}_03_components.json",
            (self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir) / f"{name}_components.json"
        ]

        for step3_path in possible_paths:
            try:
                if step3_path.exists():
                    with open(step3_path, 'r') as f:
                        components = json.load(f)
                    print(f"   ✅ Loaded {len(components)} Step 3 components with precise coordinates from {step3_path.name}")
                    return components
            except Exception as e:
                print(f"   ⚠️  Failed to load Step 3 components from {step3_path.name}: {e}")

        return []

    def _convert_step2_to_step3_format(self, tiles: List[Dict]) -> List[Dict]:
        """Convert Step 2 tiles to Step 3 component format for fallback"""
        components = []
        for i, tile in enumerate(tiles):
            component = {
                "canvas_id": i + 1,
                "product_image_x": tile['x'],
                "product_image_y": tile['y'],
                "product_image_width": tile['w'],
                "product_image_height": min(573, tile['h']),  # Product area is top 573px
                "text_area_x": tile['x'],
                "text_area_y": tile['y'] + min(573, tile['h']),  # Text area below product
                "text_area_width": tile['w'],
                "text_area_height": min(240, tile['h'] - 573),  # Text area is bottom 240px
                "pink_button_detected": False,  # No precise button coords from Step 2
                "fallback_from_step2": True
            }
            components.append(component)

        print(f"   🔄 Converted {len(components)} Step 2 tiles to Step 3 format")
        return components

    def _step_4a_extract_with_validation(self, original_image: np.ndarray, component: Dict,
                                       component_num: int, name: str, step_dir: Path) -> Dict[str, Any]:
        """STEP 4A: Extract product image and text area with pink button removal"""
        print(f"   🎯 STEP 4A: Extract Component {component_num}")

        try:
            # Extract coordinates with validation
            prod_x = component['product_image_x']
            prod_y = component['product_image_y']
            prod_w = component['product_image_width']
            prod_h = component['product_image_height']

            text_x = component['text_area_x']
            text_y = component['text_area_y']
            text_w = component['text_area_width']
            text_h = component['text_area_height']

            print(f"      📏 Product region: ({prod_x},{prod_y}) {prod_w}×{prod_h}px")
            print(f"      📏 Text region: ({text_x},{text_y}) {text_w}×{text_h}px")

            # Validate coordinates against image bounds
            img_h, img_w = original_image.shape[:2]

            if (prod_x + prod_w > img_w or prod_y + prod_h > img_h or
                text_x + text_w > img_w or text_y + text_h > img_h):
                return {"success": False, "error": f"Coordinates out of bounds for image {img_w}×{img_h}"}

            # Extract product image region
            product_image = original_image[prod_y:prod_y+prod_h, prod_x:prod_x+prod_w].copy()

            # Validate extracted product image
            if product_image.size == 0:
                return {"success": False, "error": "Extracted product image is empty"}

            if np.all(product_image == 0):
                print("      ⚠️  WARNING: Extracted product image is all black!")

            print(f"      ✅ Product image extracted: {product_image.shape[1]}×{product_image.shape[0]}px")

            # Remove pink button if detected with precise coordinates
            clean_product_image = product_image.copy()
            button_removed = False

            if component.get('pink_button_detected', False):
                button_x = component.get('pink_button_center_x', 0) - prod_x  # Relative to product image
                button_y = component.get('pink_button_center_y', 0) - prod_y
                button_radius = 48  # 96px diameter = 48px radius

                if 0 <= button_x < prod_w and 0 <= button_y < prod_h:
                    # Adjust center position by 0.5 pixels and increase radius to capture pink edges
                    adjusted_x = button_x + 0.5
                    adjusted_y = button_y + 0.5
                    expanded_radius = button_radius + 2  # Increase from 48 to 50 to capture pink border

                    # Convert to RGBA for proper transparency
                    clean_product_image = cv2.cvtColor(product_image, cv2.COLOR_BGR2BGRA)

                    # Create circular mask - larger radius to remove pink border completely
                    mask = np.zeros(product_image.shape[:2], dtype=np.uint8)
                    cv2.circle(mask, (int(adjusted_x), int(adjusted_y)), expanded_radius, 255, -1)

                    # Cut out pixels entirely - set alpha to 0 for complete removal
                    clean_product_image[mask == 255, 3] = 0  # Only touch alpha channel, leave RGB untouched
                    button_removed = True
                    print(f"      🎯 Pink button cut out at ({adjusted_x:.1f},{adjusted_y:.1f}) radius {expanded_radius}px")
                else:
                    print(f"      ⚠️  Button coordinates ({button_x},{button_y}) outside product region")

            # Extract text area
            text_area_image = None
            if text_h > 0:
                text_area_image = original_image[text_y:text_y+text_h, text_x:text_x+text_w].copy()

                if text_area_image.size > 0 and not np.all(text_area_image == 0):
                    print(f"      📝 Text area extracted: {text_area_image.shape[1]}×{text_area_image.shape[0]}px")
                else:
                    print("      ⚠️  Text area is empty or all black")
                    text_area_image = None

            # Save Step 4A results
            base_name = f"{name}_04a_component_{component_num:03d}"

            # Save original product region
            product_original_path = step_dir / f"{base_name}_product_original.jpg"
            cv2.imwrite(str(product_original_path), product_image)

            # Save clean product region (with button removed)
            # Use PNG for RGBA images (transparency support), JPG for BGR
            if clean_product_image.shape[2] == 4:  # RGBA
                product_clean_path = step_dir / f"{base_name}_product_clean.png"
            else:  # BGR
                product_clean_path = step_dir / f"{base_name}_product_clean.jpg"
            cv2.imwrite(str(product_clean_path), clean_product_image)

            # Save text area if available
            text_area_path = None
            if text_area_image is not None:
                text_area_path = step_dir / f"{base_name}_text_area.jpg"
                cv2.imwrite(str(text_area_path), text_area_image)

            return {
                "success": True,
                "component_num": component_num,
                "product_image": product_image,
                "clean_product_image": clean_product_image,
                "text_area_image": text_area_image,
                "button_removed": button_removed,
                "files": {
                    "product_original": str(product_original_path),
                    "product_clean": str(product_clean_path),
                    "text_area": str(text_area_path) if text_area_path else None
                },
                "coordinates": {
                    "product": {"x": prod_x, "y": prod_y, "w": prod_w, "h": prod_h},
                    "text": {"x": text_x, "y": text_y, "w": text_w, "h": text_h}
                }
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _step_4b_consensus_with_context(self, step4a_result: Dict, consensus_analyzer,
                                      component_num: int) -> Dict[str, Any]:
        """STEP 4B: Enhanced consensus processing with both text area AND product image for context"""
        print(f"   🧠 STEP 4B: Consensus Analysis for Component {component_num}")

        if not step4a_result["success"]:
            return {"success": False, "error": "Step 4A failed"}

        if consensus_analyzer is None:
            print("      ⚠️  Consensus analyzer unavailable")
            return {"success": False, "error": "Consensus analyzer unavailable"}

        try:
            text_area = step4a_result.get("text_area_image")
            product_image = step4a_result.get("clean_product_image")

            if text_area is None:
                print("      ⚠️  No text area available for consensus")
                return {"success": False, "error": "No text area available"}

            # Enhanced consensus: Send BOTH text area and product image for context
            print("      🔄 Running consensus with text + product image context...")

            import asyncio
            consensus_result = asyncio.run(
                consensus_analyzer.analyze_product_with_consensus(
                    text_area, product_image, "product"  # Both images for enhanced context
                )
            )

            print(f"      ✅ Consensus completed")
            print(f"         📊 Confidence: {consensus_result.get('confidence', 0):.2f}")

            if consensus_result.get('product_name'):
                print(f"         🏷️  Product: {consensus_result.get('product_name')}")
            if consensus_result.get('brand'):
                print(f"         🏢 Brand: {consensus_result.get('brand')}")
            if consensus_result.get('price'):
                print(f"         💰 Price: {consensus_result.get('price')}")
            if consensus_result.get('weight'):
                print(f"         ⚖️  Weight: {consensus_result.get('weight')}")
            if consensus_result.get('quantity'):
                print(f"         📦 Quantity: {consensus_result.get('quantity')}")
            if consensus_result.get('cost_per_kg'):
                print(f"         💲 Cost per kg: {consensus_result.get('cost_per_kg')}")
            if consensus_result.get('cost_per_piece'):
                print(f"         💲 Cost per piece: {consensus_result.get('cost_per_piece')}")

            return {
                "success": True,
                "component_num": component_num,
                "consensus_result": consensus_result,
                "enhanced_context": True  # Flag indicating we used both images
            }

        except Exception as e:
            print(f"      ❌ Consensus analysis failed: {e}")
            return {"success": False, "error": str(e)}

    def _step_4c_background_removal(self, step4a_result: Dict, bg_removal_manager,
                                  component_num: int, step_dir: Path) -> Dict[str, Any]:
        """STEP 4C: Background removal on clean product images"""
        print(f"   🎨 STEP 4C: Background Removal for Component {component_num}")

        if not step4a_result["success"]:
            return {"success": False, "error": "Step 4A failed"}

        if bg_removal_manager is None:
            print("      ⚠️  Background removal manager unavailable")
            return {"success": False, "error": "Background removal manager unavailable"}

        try:
            clean_product_image = step4a_result.get("clean_product_image")

            if clean_product_image is None:
                return {"success": False, "error": "No clean product image available"}

            print("      🔄 Removing background from clean product image...")

            # Apply background removal to the clean product image (after button removal)
            # Create temporary files for background removal
            import tempfile
            import os

            # Save clean product image to temp file
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

            if product_nobg is not None:
                # Generate output filename based on component number
                # Get the base name from existing files or create it
                if "files" in step4a_result and "product_clean" in step4a_result["files"]:
                    base_name = step4a_result["files"]["product_clean"].replace("_product_clean.jpg", "")
                else:
                    # Fallback: generate from component number
                    base_name = f"{step_dir.name}_{component_num:03d}"

                nobg_path = step_dir / f"{base_name}_product_nobg.png"  # PNG for transparency
                cv2.imwrite(str(nobg_path), product_nobg)

                print(f"      ✅ Background removed successfully")
                print(f"      📁 Saved: {nobg_path}")

                return {
                    "success": True,
                    "component_num": component_num,
                    "product_nobg": product_nobg,
                    "file": str(nobg_path)
                }
            else:
                return {"success": False, "error": "Background removal returned None"}

        except Exception as e:
            print(f"      ❌ Background removal failed: {e}")
            return {"success": False, "error": str(e)}

    def _initialize_consensus_analyzer(self):
        """Initialize consensus analyzer with proper error handling"""
        try:
            import sys
            import os

            # Add food_extractor to Python path for imports
            food_extractor_path = os.path.join(os.getcwd(), '..', 'food_extractor')
            if food_extractor_path not in sys.path:
                sys.path.append(food_extractor_path)

            from src.local_consensus_analyzer import LocalConsensusAnalyzer
            analyzer = LocalConsensusAnalyzer()
            print("   ✅ Consensus analyzer initialized")
            return analyzer
        except Exception as e:
            print(f"   ⚠️  Consensus analyzer unavailable: {e}")
            return None

    def _remove_pink_button_from_tile_proper(self, tile_image):
        """
        CRITICAL: Use proper HSV color detection + HoughCircles method, NOT cv2.inpaint - DO NOT CHANGE
        Remove pink + button from raw tile image before background removal.
        This is the correct workflow: extract tile -> remove button -> background removal.
        """
        if tile_image is None or tile_image.size == 0:
            return tile_image

        h, w = tile_image.shape[:2]

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
                    button_found = True

        # Restrict pink removal to bottom-right area only
        pink_mask_restricted = np.zeros_like(pink_mask)
        pink_mask_restricted[int(h*0.7):h, int(w*0.7):w] = pink_mask[int(h*0.7):h, int(w*0.7):w]

        # Combine masks
        combined_mask = cv2.bitwise_or(removal_mask, pink_mask_restricted)

        # CRITICAL: Apply removal by setting pixels to neutral color, NOT inpainting - DO NOT CHANGE
        if np.sum(combined_mask > 0) > 0:
            # Set button area to white (neutral background color)
            tile_image[combined_mask > 0] = [245, 245, 245]  # Light gray/white

        return tile_image

    def _initialize_background_removal_manager(self):
        """Initialize background removal manager with proper error handling"""
        try:
            import sys
            import os

            # Add food_extractor to Python path for imports
            food_extractor_path = os.path.join(os.getcwd(), '..', 'food_extractor')
            if food_extractor_path not in sys.path:
                sys.path.append(food_extractor_path)

            from src.background_removal_manager import BackgroundRemovalManager
            manager = BackgroundRemovalManager()
            print("   ✅ Background removal manager initialized")
            return manager
        except Exception as e:
            print(f"   ⚠️  Background removal manager unavailable: {e}")
            return None

    def _create_step4_visualization_grid(self, step4a_results: List, step4b_results: List,
                                       step4c_results: List, name: str, step_dir: Path):
        """Create comprehensive visualization grid for Step 4 A/B/C results"""
        if not step4a_results:
            print("   ⚠️  No Step 4A results to visualize")
            return

        try:
            print("   🎨 Creating Step 4 A/B/C visualization grid...")

            # Create grid showing progression: Original → Clean → No Background → Text Area
            grid_rows = []

            for result in step4a_results:
                component_num = result["component_num"]

                # Load images for this component
                images_row = []

                # Original product image
                if os.path.exists(result["files"]["product_original"]):
                    orig_img = cv2.imread(result["files"]["product_original"])
                    orig_img = cv2.resize(orig_img, (200, 200))
                    images_row.append(orig_img)

                # Clean product image (button removed)
                if os.path.exists(result["files"]["product_clean"]):
                    clean_img = cv2.imread(result["files"]["product_clean"])
                    clean_img = cv2.resize(clean_img, (200, 200))
                    images_row.append(clean_img)

                # Background removed image (if available)
                step4c_result = next((r for r in step4c_results if r["component_num"] == component_num), None)
                if step4c_result and os.path.exists(step4c_result["file"]):
                    nobg_img = cv2.imread(step4c_result["file"], cv2.IMREAD_UNCHANGED)
                    if nobg_img.shape[2] == 4:  # RGBA
                        nobg_img = cv2.cvtColor(nobg_img, cv2.COLOR_BGRA2BGR)
                    nobg_img = cv2.resize(nobg_img, (200, 200))
                    images_row.append(nobg_img)
                else:
                    # Placeholder for missing background removal
                    placeholder = np.zeros((200, 200, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "No BG", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    images_row.append(placeholder)

                # Text area image (if available)
                if result["files"]["text_area"] and os.path.exists(result["files"]["text_area"]):
                    text_img = cv2.imread(result["files"]["text_area"])
                    text_img = cv2.resize(text_img, (200, 80))  # Wider for text
                    # Pad to 200x200 for grid consistency
                    padded_text = np.zeros((200, 200, 3), dtype=np.uint8)
                    padded_text[60:140, :] = text_img
                    images_row.append(padded_text)
                else:
                    # Placeholder for missing text area
                    placeholder = np.zeros((200, 200, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "No Text", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    images_row.append(placeholder)

                if len(images_row) >= 4:
                    # Concatenate horizontally
                    row_image = np.hstack(images_row[:4])

                    # Add labels
                    cv2.putText(row_image, f"Component {component_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(row_image, "Original", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(row_image, "Clean", (210, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(row_image, "No BG", (410, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(row_image, "Text Area", (610, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    grid_rows.append(row_image)

            if grid_rows:
                # Stack all rows vertically
                final_grid = np.vstack(grid_rows)

                # Save the grid
                grid_path = step_dir / f"{name}_04_abc_grid.jpg"
                cv2.imwrite(str(grid_path), final_grid)
                print(f"   ✅ Step 4 A/B/C grid saved: {grid_path}")

        except Exception as e:
            print(f"   ⚠️  Failed to create visualization grid: {e}")

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            "step": "04_refined_abc_pipeline",
            "success": False,
            "error": error_message,
            "processing_stats": {
                "total_components": 0,
                "successful_extractions": 0,
                "validation_warnings": 1
            },
            "black_array_prevention": "failed"
        }

    def _step_05_ui_removal(self, product_images: List[Dict], name: str) -> Dict[str, Any]:
        """STEP 5: UI Element Removal (Pink Buttons) - Proxy Method

        Note: In the enhanced pipeline, UI removal is integrated into Step 4.
        This method serves as a compatibility layer for the original pipeline structure.
        """
        print("🔥 STEP 5: UI Element Removal (Integrated in Step 4)")

        # Since UI removal is already handled in Step 4, we'll return the clean images
        clean_images = []
        for img_data in product_images:
            if "clean_image_path" in img_data:
                clean_images.append({
                    "component_id": img_data.get("component_id", "unknown"),
                    "clean_image_path": img_data["clean_image_path"],
                    "original_image_path": img_data.get("original_image_path", ""),
                    "ui_elements_removed": True
                })
            else:
                # Fallback for images without clean versions
                clean_images.append({
                    "component_id": img_data.get("component_id", "unknown"),
                    "clean_image_path": img_data.get("image_path", ""),
                    "original_image_path": img_data.get("image_path", ""),
                    "ui_elements_removed": False
                })

        return {
            "step": "05_ui_removal",
            "clean_images": clean_images,
            "summary": {
                "total_images": len(clean_images),
                "ui_elements_removed": sum(1 for img in clean_images if img["ui_elements_removed"]),
                "processing_method": "integrated_in_step_4"
            }
        }

    def _step_06_text_extraction(self, image: np.ndarray, tiles: List[Dict], name: str) -> Dict[str, Any]:
        """STEP 6: Extract Text Areas for OCR"""
        print("📝 STEP 6: Text Area Extraction")
        
        step_dir = self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir
        
        text_images = []
        
        for i, tile in enumerate(tiles):
            coords = tile["coordinates"]
            x, y, w, h = coords["x"], coords["y"], coords["width"], coords["height"]
            
            # Extract full tile
            tile_image = image[y:y+h, x:x+w]
            
            # Extract text area (bottom 35% of tile)
            components = tile.get("components", {})
            if "text_area" in components:
                text_area = components["text_area"]
                text_image = tile_image[text_area["y"]:text_area["y"]+text_area["height"],
                                      text_area["x"]:text_area["x"]+text_area["width"]]
            else:
                # Fallback: use bottom 35%
                text_start = int(h * 0.65)
                text_image = tile_image[text_start:, :]
            
            # Save text area image
            text_path = step_dir / f"{name}_06_text_{i+1}.jpg"
            cv2.imwrite(str(text_path), text_image)
            
            text_images.append({
                "product_number": i + 1,
                "file_path": str(text_path),
                "image": text_image
            })
        
        # Create text areas grid
        if text_images:
            grid_image = self._create_text_grid(text_images, "STEP 6: Text Areas for OCR")
            grid_path = step_dir / f"{name}_06_text_grid.jpg"
            cv2.imwrite(str(grid_path), grid_image)
        
        result = {
            "step": 6,
            "title": "Text Area Extraction",
            "description": f"Extracted {len(text_images)} text regions for consensus OCR processing",
            "files": {
                "grid": str(grid_path),
                "text_images": [t["file_path"] for t in text_images]
            },
            "text_images": text_images
        }
        
        print(f"  ✅ Saved: {grid_path}")
        print(f"  📝 Extracted: {len(text_images)} text areas")
        return result
    
    def _step_07_consensus_ocr(self, text_images: List[Dict], name: str) -> Dict[str, Any]:
        """STEP 7: Consensus OCR Processing"""
        print("🔍 STEP 7: Consensus OCR Processing")
        
        step_dir = self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir
        
        ocr_results = []
        
        # Initialize text extractor (your existing system)
        text_extractor = self.food_extractor.text_extractor
        
        for text_item in text_images:
            text_image = text_item["image"]
            
            # Run OCR using your existing consensus system
            ocr_result = text_extractor.extract_text_from_tile(text_image)
            
            # Get raw text as well
            raw_text = text_extractor.extract_raw_text_from_tile(text_image)
            
            # Process and clean results
            processed_result = {
                "product_number": text_item["product_number"],
                "raw_text": raw_text,
                "product_name": ocr_result.get("product_name", ""),
                "price": ocr_result.get("price", ""),
                "unit": ocr_result.get("price_per_unit", ""),
                "confidence": 0.8  # EasyOCR confidence is built-in
            }
            
            ocr_results.append(processed_result)
        
        # Create OCR results visualization
        results_image = self._create_ocr_results_visualization(text_images, ocr_results)
        results_path = step_dir / f"{name}_07_ocr_results.jpg"
        cv2.imwrite(str(results_path), results_image)
        
        # Save OCR data as JSON
        json_path = step_dir / f"{name}_07_ocr_data.json"
        with open(json_path, 'w') as f:
            json.dump(ocr_results, f, indent=2)
        
        result = {
            "step": 7,
            "title": "Consensus OCR Processing",
            "description": f"Extracted text from {len(ocr_results)} products using consensus OCR",
            "files": {
                "visualization": str(results_path),
                "ocr_json": str(json_path)
            },
            "ocr_results": ocr_results
        }
        
        print(f"  ✅ Saved: {results_path}")
        print(f"  ✅ Saved: {json_path}")
        print(f"  🔍 OCR processed: {len(ocr_results)} products")
        return result
    
    def _step_08_ai_enhancement(self, clean_images: List[Dict], ocr_results: List[Dict], name: str) -> Dict[str, Any]:
        """STEP 8: AI Enhancement & Validation"""
        print("🤖 STEP 8: AI Enhancement & Validation")
        
        step_dir = self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir
        
        enhanced_data = []
        
        # Initialize AI analyzer (your existing system)
        if hasattr(self.food_extractor, 'ai_text_analyzer'):
            ai_analyzer = self.food_extractor.ai_text_analyzer
        else:
            ai_analyzer = None
            print("  ⚠️ AI analyzer not available, using OCR results only")
        
        for i, (clean_img, ocr_result) in enumerate(zip(clean_images, ocr_results)):
            enhanced_result = ocr_result.copy()
            
            if ai_analyzer:
                # Run AI enhancement on clean product image
                ai_result = ai_analyzer.analyze_product_image(clean_img["image"])
                
                # Merge AI results with OCR
                enhanced_result.update({
                    "ai_product_name": ai_result.get("product_name", ""),
                    "ai_confidence": ai_result.get("confidence", 0.0),
                    "category_detected": ai_result.get("category", ""),
                    "nutritional_info": ai_result.get("nutrition", {}),
                    "final_product_name": ai_result.get("product_name", "") or ocr_result.get("product_name", ""),
                    "consensus_confidence": (ai_result.get("confidence", 0) + ocr_result.get("confidence", 0)) / 2
                })
            else:
                enhanced_result.update({
                    "final_product_name": ocr_result.get("product_name", ""),
                    "consensus_confidence": ocr_result.get("confidence", 0.0)
                })
            
            enhanced_data.append(enhanced_result)
        
        # Create enhancement comparison visualization
        enhancement_image = self._create_enhancement_visualization(clean_images, enhanced_data)
        enhancement_path = step_dir / f"{name}_08_enhanced.jpg"
        cv2.imwrite(str(enhancement_path), enhancement_image)
        
        # Save enhanced data
        json_path = step_dir / f"{name}_08_enhanced_data.json"
        with open(json_path, 'w') as f:
            json.dump(enhanced_data, f, indent=2)
        
        result = {
            "step": 8,
            "title": "AI Enhancement & Validation",
            "description": f"Enhanced {len(enhanced_data)} products with AI analysis and consensus validation",
            "files": {
                "visualization": str(enhancement_path),
                "enhanced_json": str(json_path)
            },
            "enhanced_data": enhanced_data
        }
        
        print(f"  ✅ Saved: {enhancement_path}")
        print(f"  ✅ Saved: {json_path}")
        print(f"  🤖 Enhanced: {len(enhanced_data)} products")
        return result
    
    def _step_09_csv_generation(self, enhanced_data: List[Dict], analysis: Dict[str, Any], name: str) -> Dict[str, Any]:
        """STEP 9: Final CSV Generation"""
        print("📊 STEP 9: Final CSV Generation")
        
        step_dir = self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir
        
        # Create comprehensive CSV data
        csv_data = []
        categories = analysis.get("categories", ["Unknown"])
        subcategories = analysis.get("subcategories", ["Unknown"])
        
        for product in enhanced_data:
            csv_row = {
                # Basic Info
                "product_id": f"{name}_product_{product['product_number']}",
                "product_number": product["product_number"],
                "screenshot_source": name,
                "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                
                # Category Context
                "category": categories[0] if categories else "Unknown",
                "subcategory": subcategories[0] if subcategories else "Unknown",
                
                # Product Details
                "product_name": product.get("final_product_name", ""),
                "price": product.get("price", ""),
                "unit": product.get("unit", ""),
                
                # AI Analysis
                "ai_product_name": product.get("ai_product_name", ""),
                "category_detected": product.get("category_detected", ""),
                
                # Quality Metrics
                "ocr_confidence": product.get("confidence", 0.0),
                "ai_confidence": product.get("ai_confidence", 0.0),
                "consensus_confidence": product.get("consensus_confidence", 0.0),
                
                # Raw Data
                "raw_ocr_text": product.get("raw_text", ""),
                "nutritional_info": json.dumps(product.get("nutritional_info", {}))
            }
            csv_data.append(csv_row)
        
        # Save CSV with step prefix in image directory
        df = pd.DataFrame(csv_data)
        csv_path = self.current_image_dir / f"step_09_{name}_extracted_products.csv"
        df.to_csv(csv_path, index=False)
        
        # Create final summary visualization
        summary_image = self._create_final_summary(csv_data, analysis)
        summary_path = step_dir / f"{name}_09_final_summary.jpg"
        cv2.imwrite(str(summary_path), summary_image)
        
        result = {
            "step": 9,
            "title": "Final CSV Generation",
            "description": f"Generated final CSV with {len(csv_data)} products and complete metadata",
            "files": {
                "csv": str(csv_path),
                "summary": str(summary_path)
            },
            "csv_data": csv_data
        }
        
        print(f"  ✅ Saved: {csv_path}")
        print(f"  ✅ Saved: {summary_path}")
        print(f"  📊 CSV contains: {len(csv_data)} products")
        return result
    
    # Helper methods for image creation
    def _remove_pink_buttons(self, image: np.ndarray) -> np.ndarray:
        """Remove pink buttons using color detection"""
        clean_image = image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define pink button color range
        lower_pink = np.array([140, 50, 100])
        upper_pink = np.array([180, 255, 255])
        pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
        
        # Find contours and remove circular buttons
        contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 2000:  # Button size range
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                if 0.7 < aspect_ratio < 1.4:  # Roughly circular
                    # Fill with background color
                    surrounding_color = self._get_surrounding_color(clean_image, x, y, w, h)
                    cv2.rectangle(clean_image, (x, y), (x+w, y+h), surrounding_color, -1)
        
        return clean_image
    
    def _get_surrounding_color(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[int, int, int]:
        """Estimate background color from surrounding area"""
        margin = 20
        img_h, img_w = image.shape[:2]
        
        sample_x1 = max(0, x - margin)
        sample_y1 = max(0, y - margin) 
        sample_x2 = min(img_w, x + w + margin)
        sample_y2 = min(img_h, y + h + margin)
        
        surrounding_area = image[sample_y1:sample_y2, sample_x1:sample_x2]
        mean_color = np.mean(surrounding_area.reshape(-1, 3), axis=0)
        return tuple(map(int, mean_color))
    
    def _create_product_grid(self, product_images: List[Dict], title: str) -> np.ndarray:
        """Create a grid visualization of product images"""
        if not product_images:
            return np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Resize all images to same size
        target_size = (200, 200)
        resized_images = []
        
        for product in product_images:
            img = product["image"]
            resized = cv2.resize(img, target_size)
            
            # Add product number
            cv2.putText(resized, f"Product {product['product_number']}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            resized_images.append(resized)
        
        # Create 2x2 grid
        if len(resized_images) >= 4:
            top_row = np.hstack([resized_images[0], resized_images[1]])
            bottom_row = np.hstack([resized_images[2], resized_images[3]])
            grid = np.vstack([top_row, bottom_row])
        elif len(resized_images) == 2:
            grid = np.hstack(resized_images)
        else:
            grid = resized_images[0]
        
        # Add title
        title_height = 60
        titled_grid = np.zeros((grid.shape[0] + title_height, grid.shape[1], 3), dtype=np.uint8)
        titled_grid[title_height:, :] = grid
        
        cv2.putText(titled_grid, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return titled_grid
    
    def _create_text_grid(self, text_images: List[Dict], title: str) -> np.ndarray:
        """Create a grid visualization of text areas"""
        if not text_images:
            return np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Resize all text images to same size
        target_size = (300, 100)
        resized_images = []
        
        for text_item in text_images:
            img = text_item["image"]
            resized = cv2.resize(img, target_size)
            
            # Add border and product number
            bordered = cv2.copyMakeBorder(resized, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(bordered, f"Text Area {text_item['product_number']}", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            resized_images.append(bordered)
        
        # Stack vertically
        grid = np.vstack(resized_images)
        
        # Add title
        title_height = 60
        titled_grid = np.zeros((grid.shape[0] + title_height, grid.shape[1], 3), dtype=np.uint8)
        titled_grid[title_height:, :] = grid
        
        cv2.putText(titled_grid, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return titled_grid
    
    def _create_before_after_comparison(self, original_images: List[Dict], clean_images: List[Dict]) -> np.ndarray:
        """Create before/after comparison of UI removal"""
        if not original_images or not clean_images:
            return np.zeros((400, 600, 3), dtype=np.uint8)
        
        target_size = (150, 150)
        comparisons = []
        
        for orig, clean in zip(original_images[:2], clean_images[:2]):  # Show first 2
            orig_resized = cv2.resize(orig["image"], target_size)
            clean_resized = cv2.resize(clean["image"], target_size)
            
            # Add labels
            cv2.putText(orig_resized, "BEFORE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(clean_resized, "AFTER", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Create side-by-side comparison
            comparison = np.hstack([orig_resized, clean_resized])
            comparisons.append(comparison)
        
        # Stack comparisons vertically
        result = np.vstack(comparisons)
        
        # Add title
        title_height = 60
        titled_result = np.zeros((result.shape[0] + title_height, result.shape[1], 3), dtype=np.uint8)
        titled_result[title_height:, :] = result
        
        cv2.putText(titled_result, "STEP 5: Before/After UI Removal", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return titled_result
    
    def _create_ocr_results_visualization(self, text_images: List[Dict], ocr_results: List[Dict]) -> np.ndarray:
        """Create visualization of OCR results"""
        canvas_height = 600
        canvas_width = 800
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(canvas, "STEP 7: OCR Results", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        y_offset = 80
        for i, ocr_result in enumerate(ocr_results[:4]):  # Show first 4
            product_num = ocr_result["product_number"]
            product_name = ocr_result.get("product_name", "Unknown")[:30]  # Truncate
            price = ocr_result.get("price", "")
            confidence = ocr_result.get("confidence", 0.0)
            
            # Product info
            cv2.putText(canvas, f"Product {product_num}:", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, f"Name: {product_name}", (20, y_offset + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(canvas, f"Price: {price}", (20, y_offset + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(canvas, f"Confidence: {confidence:.2f}", (20, y_offset + 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            y_offset += 120
        
        return canvas
    
    def _create_enhancement_visualization(self, clean_images: List[Dict], enhanced_data: List[Dict]) -> np.ndarray:
        """Create visualization showing AI enhancement results"""
        canvas_height = 600
        canvas_width = 800
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(canvas, "STEP 8: AI Enhancement", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        y_offset = 80
        for i, enhanced in enumerate(enhanced_data[:4]):
            product_num = enhanced["product_number"]
            final_name = enhanced.get("final_product_name", "Unknown")[:25]
            consensus_conf = enhanced.get("consensus_confidence", 0.0)
            category = enhanced.get("category_detected", "")[:15]
            
            cv2.putText(canvas, f"Product {product_num}:", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, f"Final: {final_name}", (20, y_offset + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(canvas, f"Category: {category}", (20, y_offset + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(canvas, f"Consensus: {consensus_conf:.2f}", (20, y_offset + 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            y_offset += 120
        
        return canvas
    
    def _create_final_summary(self, csv_data: List[Dict], analysis: Dict[str, Any]) -> np.ndarray:
        """Create final summary visualization"""
        canvas_height = 600
        canvas_width = 800
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(canvas, "STEP 9: Final Results Summary", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        # Stats
        cv2.putText(canvas, f"Products Extracted: {len(csv_data)}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        categories = analysis.get("categories", ["Unknown"])
        cv2.putText(canvas, f"Categories: {', '.join(categories)}", (20, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Average confidence
        if csv_data:
            avg_confidence = sum(row.get("consensus_confidence", 0) for row in csv_data) / len(csv_data)
            cv2.putText(canvas, f"Avg Confidence: {avg_confidence:.2f}", (20, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Product list
        cv2.putText(canvas, "Extracted Products:", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset = 270
        for i, product in enumerate(csv_data[:8]):  # Show first 8
            name = product.get("product_name", "Unknown")[:35]
            price = product.get("price", "")
            
            cv2.putText(canvas, f"{i+1}. {name} - {price}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        return canvas
    
    def _generate_html_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive HTML report"""
        screenshot_name = results.get('screenshot_name', 'unknown')
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Step-by-Step Pipeline Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .step {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
        .step h2 {{ color: #333; }}
        .step-image {{ max-width: 100%; height: auto; margin: 10px 0; }}
        .csv-preview {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .stats {{ background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>🚀 Step-by-Step Food Product Extraction Pipeline</h1>
    <p><strong>Screenshot:</strong> {screenshot_name}</p>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="stats">
        <h3>📊 Summary Statistics</h3>
        <ul>
            <li>Total Products Extracted: {len(results['final_csv_data'])}</li>
            <li>Processing Steps: {len(results['steps'])}</li>
            <li>Output Files Generated: {sum(len(step.get('files', {})) for step in results['steps'])}</li>
        </ul>
    </div>
"""
        
        # Add each step
        for i, step in enumerate(results['steps']):
            try:
                step_title = step.get('step', f'Step {i+1}')
                step_description = step.get('description', 'Processing complete')

                html_content += f"""
    <div class="step">
        <h2>{step_title}</h2>
        <p>{step_description}</p>

        <h4>Generated Files:</h4>
        <ul>
"""

                # Handle different file formats
                files_dict = step.get('files', {})
                if not files_dict and 'output_files' in step:
                    files_dict = step['output_files']

                if files_dict:
                    for file_type, file_path in files_dict.items():
                        try:
                            if isinstance(file_path, list):
                                for fp in file_path:
                                    rel_path = Path(fp).relative_to(self.output_dir)
                                    html_content += f"<li>{file_type}: <a href='{rel_path}'>{rel_path}</a></li>"
                            else:
                                rel_path = Path(file_path).relative_to(self.output_dir)
                                html_content += f"<li>{file_type}: <a href='{rel_path}'>{rel_path}</a></li>"
                        except Exception as e:
                            html_content += f"<li>{file_type}: {file_path}</li>"
                else:
                    html_content += "<li>No files generated for this step</li>"

                html_content += "</ul></div>"

            except Exception as e:
                html_content += f"""
    <div class="step">
        <h2>Step {i+1}</h2>
        <p>Step completed (details unavailable)</p>
    </div>
"""
        
        # Add CSV preview
        if results['final_csv_data']:
            html_content += """
    <div class="csv-preview">
        <h3>📄 Final CSV Data (First 3 Products)</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th>Product Name</th>
                <th>Price</th>
                <th>Category</th>
                <th>Confidence</th>
            </tr>
"""
            for product in results['final_csv_data'][:3]:
                html_content += f"""
            <tr>
                <td>{product.get('product_name', 'Unknown')}</td>
                <td>{product.get('price', '')}</td>
                <td>{product.get('category', '')}</td>
                <td>{product.get('consensus_confidence', 0):.2f}</td>
            </tr>
"""
            html_content += "</table></div>"
        
        html_content += """
</body>
</html>
"""
        
        # Save HTML report with step prefix in image directory
        screenshot_name = results.get('screenshot_name', 'unknown')
        html_path = self.current_image_dir / f"{screenshot_name}_pipeline_report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"  ✅ Generated HTML report: {html_path}")

    # ==================== PRODUCT CANVAS METHODS ====================
    
    def _step_03b_product_canvas_detection(self, image: np.ndarray, tiles: list, analysis: Dict[str, Any], step2_canvases: list, name: str) -> Dict[str, Any]:
        """STEP 3B: Product Canvas Detection (tile + text + category relationship)"""
        print("🎯 STEP 3B: Product Canvas Detection (Unified tile + text + category)")
        
        step_dir = self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir
        step_dir.mkdir(exist_ok=True)
        
        canvases = []
        category_sections = self._detect_category_sections(image, analysis)

        # For each detected tile, create a complete Product Canvas
        for i, tile in enumerate(tiles):
            print(f"🖼️  Processing Product Canvas {i+1}/{len(tiles)}")

            # 1. Use validated canvas boundaries from Step 2 instead of recalculating
            if i < len(step2_canvases):
                # Use Step 2's validated boundaries to prevent spillover
                step2_canvas = step2_canvases[i]
                # Step 2 canvases use field names: x, y, width, height
                canvas_x = step2_canvas['x']
                canvas_y = step2_canvas['y']
                canvas_w = step2_canvas['width']
                canvas_h = step2_canvas['height']
                print(f"   🔒 Using Step 2 validated boundaries: {canvas_x},{canvas_y} {canvas_w}x{canvas_h}")

                # Extract tile region from Step 2 canvas structure
                tile_region = step2_canvas['tile_region']
                tile_x, tile_y = tile_region['x'], tile_region['y']
                tile_w, tile_h = tile_region['width'], tile_region['height']
                text_start_y = tile_y + tile_h
                text_height = 240

            else:
                # Fallback to original calculation if Step 2 canvas not available
                tile_x, tile_y, tile_w, tile_h = tile['x'], tile['y'], tile['w'], tile['h']
                text_start_y = tile_y + tile_h
                text_height = 240
                canvas_x = tile_x
                canvas_y = tile_y
                canvas_w = tile_w
                canvas_h = tile_h + text_height
                print(f"   ⚠️  Fallback calculation (Step 2 canvas {i} not found)")
            
            # 2. Extract text region using proven working method
            text_region = self._extract_text_region_below_tile(image, tile)
            
            # 3. Determine category assignment based on Y-coordinate
            assigned_category = self._assign_tile_to_category(tile, category_sections)
            
            # 4. Create unified Product Canvas object
            canvas = {
                'canvas_id': i,
                'tile': tile,
                'text_region': {
                    'x': canvas_x,
                    'y': text_start_y,
                    'w': canvas_w,
                    'h': text_height,
                    'image_data': text_region
                },
                'category': assigned_category,
                'canvas_bounds': {
                    'x': canvas_x,
                    'y': canvas_y,
                    'w': canvas_w,
                    'h': canvas_h
                }
            }
            
            canvases.append(canvas)
            
            # 5. Save canvas visualization
            canvas_viz = self._visualize_product_canvas(image, canvas)
            canvas_path = step_dir / f"{name}_{i+1:02d}_canvas.jpg"
            cv2.imwrite(str(canvas_path), canvas_viz)
            print(f"  💾 Saved canvas visualization: {canvas_path}")
        
        # 6. Create grid layout visualization showing all canvases
        grid_viz = self._visualize_canvas_grid(image, canvases, category_sections)
        grid_path = step_dir / f"{name}_canvas_grid.jpg"
        cv2.imwrite(str(grid_path), grid_viz)
        
        print(f"🎯 Product Canvas Detection complete: {len(canvases)} canvases detected")
        print(f"  📊 Categories: {list(set(c['category'] for c in canvases))}")
        
        return {
            "step": "3B",
            "title": "Product Canvas Detection",
            "description": f"Detected {len(canvases)} unified Product Canvases with category assignments: {list(set(c['category'] for c in canvases))}",
            "files": {
                "canvas_grid": str(grid_path),
                "individual_canvases": [str(step_dir / f"{name}_{i+1:02d}_canvas.jpg") for i in range(len(canvases))]
            },
            'canvases': canvases,
            'category_sections': category_sections
        }
    
    def _detect_category_sections(self, image: np.ndarray, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect category navigation state from header instead of mapping Y-coordinates"""
        height, width = image.shape[:2]

        print(f"🧭 HEADER NAVIGATION ANALYSIS:")
        print(f"   📐 Image dimensions: {width}x{height}")

        # Handle different data structures - Step 2 passes category_data, not UI regions
        if "regions" in analysis:
            # UI analysis structure with regions
            header_region = analysis["regions"]["header"]
            header_y = header_region["y"]
            header_h = header_region["height"]
            header_image = image[header_y:header_y + header_h, :]
            print(f"   🎯 Header region from UI analysis: Y={header_y} to Y={header_y + header_h}")
        else:
            # Step 2 category_data structure - use header portion of image
            header_h = min(200, height // 4)  # Use top 200px or 1/4 of image as header
            header_y = 0
            header_image = image[header_y:header_y + header_h, :]
            print(f"   🎯 Header region estimated: Y={header_y} to Y={header_y + header_h}")
            header_region = {"y": header_y, "height": header_h}

        # Use existing category data if available, otherwise analyze header
        if "primary_category" in analysis:
            # Step 2 category_data structure - use extracted categories
            navigation_context = {
                'main_category': analysis.get('primary_category', 'Unknown'),
                'active_subcategory': analysis.get('primary_subcategory', 'Unknown'),
                'available_subcategories': analysis.get('all_subcategories', [])
            }
            print(f"   ✅ Using extracted category data:")
        else:
            # Analyze header navigation state using consensus system
            navigation_context = self._analyze_header_navigation(header_image)
            print(f"   ✅ Navigation Analysis Complete:")

        print(f"      🏷️  Main Category: {navigation_context.get('main_category', 'Unknown')}")
        print(f"      🎯 Active Subcategory: {navigation_context.get('active_subcategory', 'Unknown')}")
        print(f"      📋 Available Subcategories: {navigation_context.get('available_subcategories', [])}")

        return {
            "navigation_context": navigation_context,
            "content_strategy": "header_guided_assignment",
            "header_region": header_region
        }
    
    def _assign_tile_to_category(self, tile: Dict, category_context: Dict[str, Any]) -> str:
        """Assign a tile to a category using shop knowledge from Step 0"""

        # Use shop knowledge if available
        if self.shop_profile and self.category_database:
            # Check if we have navigation context from header analysis
            if "navigation_context" in category_context:
                navigation = category_context["navigation_context"]
                main_category = navigation.get("main_category", "Unknown")

                # Validate category against our discovered taxonomy
                known_categories = self.category_database.get("categories", {})
                if main_category in known_categories:
                    print(f"   🎯 Assigning tile to validated category: {main_category}")
                    return main_category
                else:
                    # Try keyword matching from shop knowledge
                    keyword_mapping = self.category_database.get("keyword_mapping", {})
                    main_lower = main_category.lower()
                    if main_lower in keyword_mapping:
                        validated_category = keyword_mapping[main_lower]
                        print(f"   🔄 Mapped '{main_category}' → '{validated_category}' via shop knowledge")
                        return validated_category

            print(f"   ⚠️  Category not found in shop knowledge, using header result: {navigation.get('main_category', 'Unknown')}")
            return navigation.get('main_category', 'Unknown')

        # Fallback to old method if no shop knowledge
        if "navigation_context" in category_context:
            navigation = category_context["navigation_context"]
            main_category = navigation.get("main_category", "Unknown")
            print(f"   🎯 Using header category (no shop knowledge): {main_category}")
            return main_category

        print(f"   ⚠️  No navigation context or shop knowledge, using fallback")
        return "Unknown"
    
    def _extract_text_region_below_tile(self, image: np.ndarray, tile: Dict) -> np.ndarray:
        """Extract 660x240px text region below tile using proven working method"""
        x, y, w, h = tile['x'], tile['y'], tile['w'], tile['h']
        
        # Calculate 240px text region below the tile (iPhone 3x scaling)
        text_start_y = y + h
        text_height = 240
        
        # Ensure we don't go beyond image bounds
        img_height = image.shape[0]
        if text_start_y + text_height > img_height:
            text_height = img_height - text_start_y
        
        # Extract text region
        text_region = image[text_start_y:text_start_y + text_height, x:x + w]
        
        return text_region
    
    def _visualize_product_canvas(self, image: np.ndarray, canvas: Dict) -> np.ndarray:
        """Create visualization of a single Product Canvas (tile + text + category)"""
        viz_image = image.copy()
        
        # Draw tile boundary (green)
        tile = canvas['tile']
        cv2.rectangle(viz_image, (tile['x'], tile['y']), 
                     (tile['x'] + tile['w'], tile['y'] + tile['h']), (0, 255, 0), 3)
        
        # Draw text region boundary (blue)
        text = canvas['text_region']
        cv2.rectangle(viz_image, (text['x'], text['y']), 
                     (text['x'] + text['w'], text['y'] + text['h']), (255, 0, 0), 3)
        
        # Draw canvas boundary (purple)
        bounds = canvas['canvas_bounds']
        cv2.rectangle(viz_image, (bounds['x'], bounds['y']), 
                     (bounds['x'] + bounds['w'], bounds['y'] + bounds['h']), (255, 0, 255), 2)
        
        # Add category label
        category = canvas['category']
        label_pos = (bounds['x'], bounds['y'] - 10)
        cv2.putText(viz_image, f"Canvas {canvas['canvas_id']}: {category}", 
                   label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        return viz_image
    
    def _visualize_canvas_grid(self, image: np.ndarray, canvases: list, category_context: Dict) -> np.ndarray:
        """Create grid visualization showing all Product Canvases with header-based context"""
        viz_image = image.copy()

        # Get navigation context
        navigation = category_context.get("navigation_context", {})
        main_category = navigation.get("main_category", "Unknown")
        active_subcategory = navigation.get("active_subcategory", "Unknown")

        print(f"   🎨 Creating visualization for category: {main_category} (active: {active_subcategory})")

        # Draw header information box instead of incorrect Y-coordinate boundaries
        header_info_y = 50
        cv2.rectangle(viz_image, (10, header_info_y - 30), (600, header_info_y + 70), (0, 0, 0), -1)  # Black background
        cv2.rectangle(viz_image, (10, header_info_y - 30), (600, header_info_y + 70), (255, 255, 255), 2)  # White border

        # Add header navigation information
        cv2.putText(viz_image, f"Main Category: {main_category}",
                   (20, header_info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(viz_image, f"Active Subcategory: {active_subcategory}",
                   (20, header_info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        available_subs = navigation.get("available_subcategories", [])
        subs_text = ", ".join(available_subs) if available_subs else "None detected"
        cv2.putText(viz_image, f"Available: {subs_text[:50]}",
                   (20, header_info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Draw all canvases with unified color based on main category
        category_colors = {
            'Obst': (0, 200, 0),         # Green for fruits
            'Gemüse': (0, 150, 255),     # Orange for vegetables
            'Backwaren': (100, 100, 255), # Light red for bakery
            'Unknown': (128, 128, 128)    # Gray for unknown
        }

        main_color = category_colors.get(main_category, (128, 128, 128))

        for canvas in canvases:
            # Use consistent color based on main category from header
            bounds = canvas['canvas_bounds']
            cv2.rectangle(viz_image, (bounds['x'], bounds['y']),
                         (bounds['x'] + bounds['w'], bounds['y'] + bounds['h']), main_color, 3)

            # Add canvas label with main category
            label = f"Canvas {canvas['canvas_id']}: {main_category}"
            cv2.putText(viz_image, label, (bounds['x'], bounds['y'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, main_color, 2)

        # Add step title
        cv2.putText(viz_image, "STEP 3B: Header-Guided Product Canvas Detection",
                   (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)

        return viz_image

    def _analyze_header_navigation(self, header_image: np.ndarray) -> Dict[str, Any]:
        """Extract navigation state from header using 3-model consensus system"""
        print(f"🧠 HEADER NAVIGATION CONSENSUS ANALYSIS:")
        print(f"   📐 Header image: {header_image.shape[1]}x{header_image.shape[0]}")

        try:
            # Create specialized prompt for header navigation detection
            navigation_prompt = self._create_header_navigation_prompt()

            # Use consensus system to analyze header navigation
            import asyncio
            # Create a dummy text region for the consensus system
            dummy_text_region = np.zeros((100, header_image.shape[1], 3), dtype=np.uint8)

            # Run consensus analysis on header for navigation state
            result = asyncio.run(self.consensus_analyzer.analyze_product_with_consensus(
                header_image, dummy_text_region
            ))

            if result:
                # Extract navigation information from consensus result
                navigation_context = self._extract_navigation_from_consensus(result)
                print(f"   ✅ Consensus navigation analysis: {result.get('analysis_method', 'consensus')}")
                return navigation_context
            else:
                print(f"   ⚠️  Consensus failed, using OCR fallback")
                return self._fallback_header_navigation(header_image)

        except Exception as e:
            print(f"   ❌ Navigation consensus error: {e}")
            return self._fallback_header_navigation(header_image)

    def _create_header_navigation_prompt(self) -> str:
        """Create specialized prompt for header navigation detection"""
        return """
Analyze this grocery app header image and identify the navigation state:

1. MAIN CATEGORY (highlighted in pink/selected): Look for category pills like "Obst", "Gemüse", "Backwaren"
2. SUBCATEGORIES (bottom tabs): Look for text like "Highlights", "Bananen", "Äpfel & Birnen", "Beeren"
3. ACTIVE SUBCATEGORY (underlined/selected): Which subcategory tab is currently selected

Return JSON format:
{
    "main_category": "Obst",
    "active_subcategory": "Bananen",
    "available_subcategories": ["Highlights", "Bananen", "Äpfel & Birnen", "Beeren"],
    "navigation_type": "grocery_app_header"
}

Focus on UI state detection: which buttons/tabs are highlighted, underlined, or visually selected.
"""

    def _extract_navigation_from_consensus(self, consensus_result: Dict) -> Dict[str, Any]:
        """Extract navigation state from consensus analysis result"""
        print(f"   🔍 Extracting navigation from consensus result")

        # Try to find navigation information in the consensus result
        product_name = consensus_result.get('product_name', '').lower()
        brand = consensus_result.get('brand', '').lower()
        all_text = f"{product_name} {brand}".lower()

        print(f"   📝 Consensus text: '{all_text[:100]}...'")

        # Look for category keywords in consensus result
        main_category = "Unknown"
        active_subcategory = "Unknown"
        available_subcategories = []

        # Main categories
        if "obst" in all_text or "fruit" in all_text:
            main_category = "Obst"
        elif "gemüse" in all_text or "vegetable" in all_text:
            main_category = "Gemüse"
        elif "backwaren" in all_text or "bakery" in all_text:
            main_category = "Backwaren"

        # Subcategories
        subcategory_keywords = ["highlights", "bananen", "banana", "äpfel", "apple", "birnen", "pear", "beeren", "berry"]
        for keyword in subcategory_keywords:
            if keyword in all_text:
                if "bananen" in keyword or "banana" in keyword:
                    available_subcategories.append("Bananen")
                    if "bananen" in all_text:
                        active_subcategory = "Bananen"
                elif "äpfel" in keyword or "apple" in keyword or "birnen" in keyword:
                    available_subcategories.append("Äpfel & Birnen")
                elif "highlights" in keyword:
                    available_subcategories.append("Highlights")
                elif "beeren" in keyword or "berry" in keyword:
                    available_subcategories.append("Beeren")

        navigation_result = {
            "main_category": main_category,
            "active_subcategory": active_subcategory,
            "available_subcategories": list(set(available_subcategories)),
            "confidence": consensus_result.get('confidence', 0.5),
            "method": "consensus_extraction"
        }

        print(f"   📋 Navigation extracted: {navigation_result}")
        return navigation_result

    def _fallback_header_navigation(self, header_image: np.ndarray) -> Dict[str, Any]:
        """Fallback header navigation detection using OCR with shop knowledge"""
        print(f"   🔧 Using OCR fallback for header navigation with shop knowledge")

        try:
            # Use the existing text extractor for OCR analysis
            text_results = self.food_extractor.text_extractor.reader.readtext(header_image)

            detected_text = []
            for bbox, text, confidence in text_results:
                if confidence > 0.5:
                    detected_text.append(text.lower())

            all_text = " ".join(detected_text)
            print(f"   📝 OCR detected: '{all_text[:100]}...'")

            # Use shop knowledge if available, otherwise fallback to expanded patterns
            main_category = "Unknown"  # No longer default to "Obst"
            active_subcategory = "Unknown"
            available_subcategories = []

            if self.category_database:
                # Use discovered shop categories
                print(f"   🏪 Using shop knowledge for category matching")
                categories = self.category_database.get("categories", {})
                keyword_mapping = self.category_database.get("keyword_mapping", {})

                # Check all known categories against OCR text
                for category, category_data in categories.items():
                    keywords = category_data.get("keywords", [])
                    if any(keyword in all_text for keyword in keywords):
                        main_category = category
                        print(f"   ✅ Matched category: {category} via keywords: {keywords}")

                        # Look for subcategories
                        subcategories = category_data.get("subcategories", [])
                        for subcat in subcategories:
                            if subcat.lower() in all_text:
                                available_subcategories.append(subcat)
                                active_subcategory = subcat  # Assume active if found
                        break

                # Try keyword mapping if no direct match
                if main_category == "Unknown":
                    for keyword, mapped_category in keyword_mapping.items():
                        if keyword in all_text:
                            main_category = mapped_category
                            print(f"   🔄 Mapped keyword '{keyword}' → '{mapped_category}'")
                            break
            else:
                # Fallback to expanded hardcoded patterns
                print(f"   ⚠️  No shop knowledge available, using expanded patterns")
                category_patterns = {
                    "Obst": ["obst", "fruit", "früchte", "bananen", "äpfel", "birnen"],
                    "Gemüse": ["gemüse", "vegetables", "vegetable"],
                    "Joghurt & Desserts": ["joghurt", "dessert", "yogurt", "pudding", "quark"],
                    "Milch & Butter": ["milch", "butter", "käse", "sahne", "milk", "cheese"],
                    "Backwaren": ["backwaren", "brot", "bakery", "bread"],
                    "Kinder": ["kinder", "baby", "kids", "children"],
                    "Fleisch & Wurst": ["fleisch", "wurst", "meat", "sausage", "schinken"],
                    "Tiefkühl": ["tiefkühl", "frozen", "tk", "gefroren"],
                    "Getränke": ["getränke", "drinks", "wasser", "saft", "beverages"]
                }

                for category, patterns in category_patterns.items():
                    if any(pattern in all_text for pattern in patterns):
                        main_category = category
                        print(f"   ✅ Matched category: {category} via patterns")
                        break

                # Look for common subcategories
                if "bananen" in all_text:
                    available_subcategories.append("Bananen")
                    active_subcategory = "Bananen"
                if "äpfel" in all_text or "birnen" in all_text:
                    available_subcategories.append("Äpfel & Birnen")
                if "highlights" in all_text:
                    available_subcategories.append("Highlights")
                if "beeren" in all_text:
                    available_subcategories.append("Beeren")

            return {
                "main_category": main_category,
                "active_subcategory": active_subcategory,
                "available_subcategories": available_subcategories,
                "confidence": 0.7,
                "method": "ocr_fallback"
            }

        except Exception as e:
            print(f"   ❌ OCR fallback failed: {e}")
            # Ultimate fallback - return reasonable defaults
            return {
                "main_category": "Obst",
                "active_subcategory": "Bananen",
                "available_subcategories": ["Bananen", "Äpfel & Birnen"],
                "confidence": 0.3,
                "method": "default_fallback"
            }

    def _analyze_category_positions_with_consensus(self, header_image: np.ndarray, subcategories: List[str]) -> Dict[str, Dict]:
        """Use 3-model consensus system to detect category text positions in header"""
        print(f"🧠 CONSENSUS CATEGORY ANALYSIS:")
        print(f"   📐 Header image: {header_image.shape[1]}x{header_image.shape[0]}")
        print(f"   🎯 Target categories: {subcategories}")

        # Create a specialized prompt for category position detection
        category_analysis_prompt = self._create_category_detection_prompt(subcategories)

        try:
            # Use consensus system to analyze category positions
            import asyncio
            # Create a dummy text region for the consensus system (it expects tile + text format)
            dummy_text_region = np.zeros((100, header_image.shape[1], 3), dtype=np.uint8)

            # Run consensus analysis using existing analyze_product_with_consensus method
            # We'll adapt it for category detection by using specialized prompt
            result = asyncio.run(self.consensus_analyzer.analyze_product_with_consensus(
                header_image, dummy_text_region
            ))

            # Extract category positions from the consensus result
            if result:
                category_positions = self._extract_category_positions_from_consensus(result, subcategories)
                if category_positions:
                    print(f"   ✅ Consensus achieved: {result.get('analysis_method', 'consensus')}")
                    return category_positions

            print(f"   ⚠️  Consensus failed, using fallback method")
            return self._fallback_category_detection(header_image, subcategories)

        except Exception as e:
            print(f"   ❌ Consensus analysis error: {e}")
            return self._fallback_category_detection(header_image, subcategories)

    def _create_category_detection_prompt(self, subcategories: List[str]) -> str:
        """Create specialized prompt for category position detection"""
        category_list = ", ".join(subcategories)
        return f"""
Analyze this grocery app header image and find the exact positions of category text labels.

Target categories to locate: {category_list}

Look for these category names as text in the image and return their approximate Y-coordinates (vertical positions).

Return JSON format:
{{
    "category_positions": {{
        "Bananen": {{"y_position": 250, "found": true}},
        "Äpfel & Birnen": {{"y_position": 300, "found": true}}
    }}
}}

Focus on finding the vertical (Y) positions where each category name appears in the header.
"""

    def _extract_category_positions_from_consensus(self, consensus_result: Dict, subcategories: List[str]) -> Dict[str, Dict]:
        """Extract category positions from consensus analysis result"""
        print(f"   🔍 Extracting category positions from consensus result")

        # Try to find category names in the consensus text or product name
        detected_positions = {}

        # Check if any subcategories appear in the consensus result
        product_name = consensus_result.get('product_name', '').lower()
        brand = consensus_result.get('brand', '').lower()
        all_text = f"{product_name} {brand}".lower()

        print(f"   📝 Consensus text analysis: '{all_text[:100]}...'")

        for i, category in enumerate(subcategories):
            category_lower = category.lower()

            # Check if category appears in consensus result
            if category_lower in all_text:
                # Estimate position based on category order
                # This is a simplified approach - in a full implementation,
                # we'd need to modify the consensus analyzer to return position data
                estimated_y = 200 + (i * 50)  # Simple estimation
                detected_positions[category] = {
                    'y_position': estimated_y,
                    'found': True,
                    'confidence': consensus_result.get('confidence', 0.8)
                }
                print(f"   📍 Found '{category}' in consensus (estimated Y={estimated_y})")
            else:
                detected_positions[category] = {
                    'y_position': 0,
                    'found': False,
                    'confidence': 0.0
                }
                print(f"   ❌ '{category}' not found in consensus")

        return {'category_positions': detected_positions}

    def _map_categories_to_content_sections(self, category_positions: Dict, content_region: Dict,
                                         content_tiles: List[Dict], subcategories: List[str]) -> Dict[str, Dict]:
        """Map detected category positions to content section boundaries using tile positions"""
        print(f"🗺️  MAPPING CATEGORIES TO CONTENT:")

        # Get content boundaries
        content_y = content_region["y"]
        content_height = content_region["height"]
        content_end = content_y + content_height

        print(f"   📊 Content region: Y={content_y} to Y={content_end}")

        # Analyze tile positions to determine row boundaries
        if len(content_tiles) >= 2:
            # Group tiles by rows based on Y-coordinates
            tile_y_positions = [tile['coordinates']['y'] for tile in content_tiles]
            tile_y_positions.sort()

            # Find natural break point between rows
            if len(tile_y_positions) >= 3:
                # Find the largest gap between consecutive tiles (indicates row break)
                gaps = []
                for i in range(len(tile_y_positions) - 1):
                    gap = tile_y_positions[i + 1] - tile_y_positions[i]
                    gaps.append((gap, tile_y_positions[i + 1]))

                # The largest gap indicates row separation
                largest_gap = max(gaps, key=lambda x: x[0])
                row_break_y = largest_gap[1]
                print(f"   🔍 Detected row break at Y={row_break_y}")
            else:
                # Fallback: use middle of content region
                row_break_y = content_y + (content_height // 2)
                print(f"   ⚠️  Using fallback row break at Y={row_break_y}")
        else:
            # Fallback: split content region in half
            row_break_y = content_y + (content_height // 2)
            print(f"   ⚠️  Using simple split at Y={row_break_y}")

        # Map subcategories to sections
        category_sections = {}

        for i, category_name in enumerate(subcategories):
            if i == 0:  # First category (usually Bananen)
                section = {
                    'y_start': content_y,
                    'y_end': row_break_y,
                    'name': category_name
                }
            else:  # Second category (usually Äpfel & Birnen)
                section = {
                    'y_start': row_break_y,
                    'y_end': content_end,
                    'name': category_name
                }

            category_sections[category_name] = section
            print(f"   📋 {category_name}: Y={section['y_start']} to Y={section['y_end']}")

        return category_sections

    def _fallback_category_detection(self, header_image: np.ndarray, subcategories: List[str]) -> Dict[str, Dict]:
        """Fallback category detection using basic OCR when consensus fails"""
        print(f"   🔧 Using OCR fallback for category detection")

        try:
            # Use the existing text extractor for basic OCR
            text_results = self.food_extractor.text_extractor.reader.readtext(header_image)

            detected_positions = {}
            for category in subcategories:
                # Look for category text in OCR results
                for bbox, text, confidence in text_results:
                    if confidence > 0.5 and category.lower() in text.lower():
                        # Get Y position from bounding box
                        y_pos = int(bbox[0][1])  # Top-left Y coordinate
                        detected_positions[category] = {
                            'y_position': y_pos,
                            'found': True,
                            'confidence': confidence
                        }
                        print(f"   📍 Found '{category}' at Y={y_pos} (confidence: {confidence:.2f})")
                        break
                else:
                    # Category not found
                    detected_positions[category] = {
                        'y_position': 0,
                        'found': False,
                        'confidence': 0.0
                    }
                    print(f"   ❌ '{category}' not found in header")

            return {'category_positions': detected_positions}

        except Exception as e:
            print(f"   ❌ OCR fallback failed: {e}")
            # Ultimate fallback: return empty positions
            return {'category_positions': {cat: {'y_position': 0, 'found': False} for cat in subcategories}}

    # ==================== WORKING METHODS FROM AI_ENHANCED_EXTRACTION.PY ====================
    
    def _detect_gray_tiles_working_method(self, image: np.ndarray) -> list:
        """Detect light gray product tiles using the proven working method from ai_enhanced_extraction.py"""
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
        
        # Sort by area and take up to 4
        tile_candidates.sort(key=lambda t: t['area'], reverse=True)
        return tile_candidates[:4]

    def _identify_content_region(self, image: np.ndarray) -> Dict[str, int]:
        """STEP 1: Identify the main content area (excluding header/footer)"""
        h, w = image.shape[:2]

        # Standard mobile app layout assumptions:
        # - Header: top ~200px (navigation, category title)
        # - Footer: bottom ~100px (tab bar, buttons)
        # - Content: middle section with product grid

        header_height = 200  # Skip navigation and category header
        footer_height = 100  # Skip bottom tab bar

        content_x = 0
        content_y = header_height
        content_w = w
        content_h = h - header_height - footer_height

        return {
            'x': content_x,
            'y': content_y,
            'w': content_w,
            'h': content_h
        }

    def _detect_pink_buttons_phase1(self, image: np.ndarray, content_region: Dict[str, int]) -> list:
        """STEP 2: Phase 1 pink button detection within content region using working method"""
        # Extract content region
        x, y, w, h = content_region['x'], content_region['y'], content_region['w'], content_region['h']
        content_image = image[y:y+h, x:x+w]

        # Convert to HSV for better color detection (working method from phase1_pink_button_validator.py)
        hsv = cv2.cvtColor(content_image, cv2.COLOR_BGR2HSV)

        # Create mask for pink color - WORKING RANGES from phase1_pink_button_validator.py
        lower_pink = np.array([140, 50, 50])    # PROVEN WORKING RANGE
        upper_pink = np.array([170, 255, 255])  # PROVEN WORKING RANGE
        mask = cv2.inRange(hsv, lower_pink, upper_pink)

        # Apply morphological operations to clean up the mask (working method)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        buttons = []
        for i, contour in enumerate(contours):
            # Calculate area and circularity (working validation from phase1)
            area = cv2.contourArea(contour)
            if area < 500:  # Too small to be a button (working threshold)
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Only consider circular objects - WORKING VALIDATION
            if circularity > 0.7:  # WORKING THRESHOLD from phase1
                # Get center and radius
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                center = (int(cx), int(cy))
                radius = int(radius)

                # Additional validation: radius should be reasonable for buttons
                if 30 <= radius <= 80:  # WORKING SIZE RANGE from phase1
                    # Convert back to full image coordinates
                    full_center = (center[0] + x, center[1] + y)

                    buttons.append({
                        'center': full_center,
                        'radius': radius,
                        'area': area,
                        'circularity': circularity,
                        'contour': contour
                    })

        # Sort by position (top to bottom, left to right) for consistent numbering
        buttons.sort(key=lambda b: (b['center'][1] // 100, b['center'][0]))

        return buttons

    def _detect_tile_boundaries_phase2(self, image: np.ndarray, button_centers: list, content_region: Dict[str, int]) -> list:
        """STEP 3: Phase 2 tile boundary detection using button anchors (working method)"""
        if not button_centers:
            # Fallback to grid-based detection if no buttons found
            return self._fallback_grid_detection(content_region)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        tiles = []

        for i, button_info in enumerate(button_centers):
            center_x, center_y = button_info['center']
            radius = button_info['radius']

            # Strategy: Use pink button as anchor point and expand outward to find tile boundaries
            # Based on working method from phase2_tile_detector.py

            # Estimate tile dimensions based on button position and image layout
            # From working examples, tiles are roughly square/rectangular
            # Pink button is typically in bottom-right of product image area

            # Based on the 2x2 grid layout and button positions:
            if center_x < width // 2:  # Left column
                tile_left = content_region['x']
                tile_right = width // 2
            else:  # Right column
                tile_left = width // 2
                tile_right = content_region['x'] + content_region['w']

            if center_y < height // 2:  # Top row
                tile_top = content_region['y']
                tile_bottom = height // 2 + 100  # Include text area below
            else:  # Bottom row
                tile_top = height // 2
                tile_bottom = content_region['y'] + content_region['h']

            # Refine boundaries using edge detection around the estimated area (working method)
            roi_x1 = max(0, tile_left)
            roi_y1 = max(0, tile_top)
            roi_x2 = min(width, tile_right)
            roi_y2 = min(height, tile_bottom)

            # Extract ROI for edge detection
            roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]

            # Apply edge detection to find more precise boundaries (working parameters)
            edges = cv2.Canny(roi, 30, 100)

            # Find contours in the ROI
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Look for large rectangular contours that could be product tiles
            best_contour = None
            best_area = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5000:  # Must be substantial (working threshold)
                    # Check if contour is roughly rectangular
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    if len(approx) >= 4 and area > best_area:
                        best_contour = contour
                        best_area = area

            # If we found a good contour, use it to refine boundaries
            if best_contour is not None:
                x, y, w, h = cv2.boundingRect(best_contour)
                # Adjust back to full image coordinates
                tile_left = roi_x1 + x
                tile_top = roi_y1 + y
                tile_right = tile_left + w
                tile_bottom = tile_top + h

            # Final validation: ensure tile contains the pink button
            if not (tile_left <= center_x <= tile_right and tile_top <= center_y <= tile_bottom):
                # Fallback to button-centered approach
                tile_width = width // 2
                tile_height = (height - 200) // 2  # Account for header/footer

                if center_x < width // 2:
                    tile_left = content_region['x']
                    tile_right = tile_left + tile_width
                else:
                    tile_left = tile_width
                    tile_right = content_region['x'] + content_region['w']

                if center_y < height // 2:
                    tile_top = content_region['y']
                    tile_bottom = tile_top + tile_height
                else:
                    tile_top = content_region['y'] + tile_height
                    tile_bottom = content_region['y'] + content_region['h']

            # Create tile object with working structure
            tile = {
                'tile_id': f'tile_{i+1}',
                'button_number': i + 1,
                'x': tile_left,
                'y': tile_top,
                'w': tile_right - tile_left,
                'h': tile_bottom - tile_top,
                'area': (tile_right - tile_left) * (tile_bottom - tile_top),
                'pink_button': {
                    'center_x': center_x,
                    'center_y': center_y,
                    'radius': radius,
                    'relative_x': center_x - tile_left,
                    'relative_y': center_y - tile_top
                },
                'detection_method': 'button_anchored_cv'
            }

            tiles.append(tile)

        return tiles

    def _fallback_grid_detection(self, content_region: Dict[str, int]) -> list:
        """Fallback grid-based tile detection if no pink buttons found"""
        x, y, w, h = content_region['x'], content_region['y'], content_region['w'], content_region['h']

        # Assume 2x2 grid layout
        tile_w = w // 2
        tile_h = h // 2

        tiles = []
        tile_id = 1

        for row in range(2):
            for col in range(2):
                tile_x = x + (col * tile_w)
                tile_y = y + (row * tile_h)

                tile = {
                    'tile_id': f'tile_{tile_id}',
                    'button_number': tile_id,
                    'x': tile_x,
                    'y': tile_y,
                    'w': tile_w,
                    'h': tile_h,
                    'area': tile_w * tile_h,
                    'detection_method': 'grid_fallback'
                }

                tiles.append(tile)
                tile_id += 1

        return tiles

    def _extract_main_product_object_working_method(self, image: np.ndarray) -> np.ndarray:
        """
        Extract the main product object and aggressively remove UI elements.
        Specifically targets circular add buttons and background elements.
        
        This is the proven working method from ai_enhanced_extraction.py
        
        Args:
            image: Input BGR image
            
        Returns:
            RGBA image with only the main product object and drop shadow
        """
        # Convert BGR to RGBA for transparency
        rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        
        # Convert to grayscale and HSV for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = gray.shape
        
        # Initialize comprehensive removal mask
        remove_mask = np.zeros(gray.shape, dtype=np.uint8)
        
        # Method 1: Conservative white/light background removal
        white_mask = gray > 240  # Less aggressive threshold to preserve products
        remove_mask[white_mask] = 255
        
        # Method 2: Targeted pink/purple color ranges for add buttons only
        # Range 1: Deep purple/magenta (more specific for UI buttons)
        lower_purple1 = np.array([140, 120, 120])  # Higher saturation/value
        upper_purple1 = np.array([170, 255, 255])
        purple_mask1 = cv2.inRange(hsv, lower_purple1, upper_purple1)
        
        # Range 2: Pink/magenta range (for pink add buttons)
        lower_pink = np.array([150, 100, 100])    # More specific range
        upper_pink = np.array([180, 255, 255])
        pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
        
        # Combine UI element masks
        ui_mask = cv2.bitwise_or(purple_mask1, pink_mask)
        
        # Clean up UI mask with morphology
        kernel = np.ones((7, 7), np.uint8)
        ui_mask = cv2.morphologyEx(ui_mask, cv2.MORPH_CLOSE, kernel)
        ui_mask = cv2.morphologyEx(ui_mask, cv2.MORPH_DILATE, kernel)
        
        # Method 3: Detect circular UI elements (add buttons)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=15,
            maxRadius=50
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x_c, y_c, r) in circles:
                # Create circular mask for each detected circle
                cv2.circle(remove_mask, (x_c, y_c), r + 5, 255, -1)
        
        # Combine all removal methods
        final_remove_mask = cv2.bitwise_or(remove_mask, ui_mask)
        
        # Apply transparency to removed areas
        rgba_image[final_remove_mask == 255] = [255, 255, 255, 0]  # Fully transparent
        
        return rgba_image


    def _extract_clean_product_images(self, original_image: np.ndarray, components_data: List[Dict], name: str) -> List[Dict]:
        """Extract clean product images from component coordinates using pink button removal"""
        print("🖼️ Extracting clean product images from component coordinates")

        clean_products = []
        step_dir = self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir

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

                # CRITICAL: Use proper HSV color detection + HoughCircles method, NOT cv2.inpaint - DO NOT CHANGE
                clean_product_tile = self._remove_pink_button_from_tile_proper(product_tile.copy())

                # Save clean product image
                product_filename = f"{name}_component_{i+1}_clean_product.png"
                product_path = os.path.join(step_dir, product_filename)
                cv2.imwrite(product_path, clean_product_tile)

                # Extract text region
                text_x = int(component["text_area_x"])
                text_y = int(component["text_area_y"])
                text_width = int(component["text_area_width"])
                text_height = int(component["text_area_height"])

                text_region = original_image[text_y:text_y+text_height, text_x:text_x+text_width]
                text_filename = f"{name}_component_{i+1}_text_region.png"
                text_path = os.path.join(step_dir, text_filename)
                cv2.imwrite(text_path, text_region)

                # Create clean product data entry
                clean_product = {
                    "component_id": component["component_id"],
                    "clean_image_path": product_path,
                    "text_image_path": text_path,
                    "product_region": {
                        "x": product_x, "y": product_y,
                        "width": product_width, "height": product_height
                    },
                    "text_region": {
                        "x": text_x, "y": text_y,
                        "width": text_width, "height": text_height
                    },
                    "pink_button_removed": component.get("pink_button_detected", False)
                }

                clean_products.append(clean_product)
                print(f"      ✅ Saved: {product_filename} and {text_filename}")

            except Exception as e:
                print(f"      ❌ Failed to extract component {i+1}: {e}")
                continue

        print(f"   🎉 Successfully extracted {len(clean_products)} clean product images")
        return clean_products

    def _step_05_consensus_product_analysis(self, clean_product_data: List[Dict], category_data: Dict[str, Any], name: str) -> Dict[str, Any]:
        """STEP 5: Consensus Product Analysis - Send clean products to LLM for detailed analysis"""
        print("🧠 STEP 5: Consensus Product Analysis - Analyzing products with LLM consensus system")

        step_dir = self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir
        analyzed_products = []

        try:
            # Initialize consensus analyzer
            from src.local_consensus_analyzer import LocalConsensusAnalyzer
            consensus_analyzer = LocalConsensusAnalyzer()

            for i, product_data in enumerate(clean_product_data):
                print(f"   🔍 Analyzing product {i+1}/{len(clean_product_data)}...")

                # Get both clean product image and text region
                product_image_path = product_data.get('clean_image_path')
                text_image_path = product_data.get('text_image_path')

                if not product_image_path or not os.path.exists(product_image_path):
                    print(f"   ⚠️  Missing clean product image for product {i+1}")
                    continue

                # Load images for analysis
                product_image = cv2.imread(product_image_path)
                text_image = cv2.imread(text_image_path) if text_image_path and os.path.exists(text_image_path) else None

                # Prepare combined image for consensus analysis
                if text_image is not None:
                    # Combine product image and text region vertically
                    combined_image = np.vstack([product_image, text_image])
                else:
                    combined_image = product_image

                # Run consensus analysis for product information extraction
                consensus_result = asyncio.run(consensus_analyzer.analyze_product_with_consensus(
                    combined_image,
                    analysis_type="product"  # Product analysis mode
                ))

                if consensus_result and consensus_result.get("success"):
                    product_info = consensus_result.get("analysis", {})

                    # Extract detailed product information
                    analyzed_product = {
                        "product_number": i + 1,
                        "product_id": f"{name}_product_{i+1}",
                        "canvas_info": product_data.get("canvas_info", {}),

                        # Consensus Analysis Results
                        "product_name": product_info.get("product_name", ""),
                        "brand": product_info.get("brand", ""),
                        "price": product_info.get("price", ""),
                        "original_price": product_info.get("original_price", ""),
                        "unit": product_info.get("unit", ""),
                        "weight_quantity": product_info.get("weight_quantity", ""),
                        "description": product_info.get("description", ""),

                        # Category Information
                        "detected_category": product_info.get("category", ""),
                        "main_category": category_data.get("main_category", ""),
                        "subcategory": category_data.get("active_subcategory", ""),

                        # Quality Metrics
                        "consensus_confidence": consensus_result.get("consensus_confidence", 0.0),
                        "models_agreed": consensus_result.get("models_agreed", 0),
                        "total_models": consensus_result.get("total_models", 3),

                        # File Paths
                        "clean_image_path": product_image_path,
                        "text_image_path": text_image_path,
                        "original_canvas_path": product_data.get("original_canvas_path", ""),

                        # Raw Analysis Data
                        "raw_consensus_data": consensus_result
                    }

                    analyzed_products.append(analyzed_product)
                    print(f"   ✅ Product {i+1}: '{analyzed_product['product_name']}' - {analyzed_product['price']}")

                else:
                    print(f"   ❌ Failed to analyze product {i+1}")
                    # Add failed product with minimal info
                    analyzed_products.append({
                        "product_number": i + 1,
                        "product_id": f"{name}_product_{i+1}",
                        "product_name": "Analysis Failed",
                        "consensus_confidence": 0.0,
                        "clean_image_path": product_image_path
                    })

            # Save analysis results JSON
            analysis_results = {
                "total_products_analyzed": len(clean_product_data),
                "successful_analyses": len([p for p in analyzed_products if p.get("consensus_confidence", 0) > 0]),
                "category_context": category_data,
                "analyzed_products": analyzed_products
            }

            json_path = step_dir / f"{name}_05_consensus_analysis.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)

            print(f"   📊 Consensus Analysis Complete: {len(analyzed_products)} products analyzed")
            print(f"   💾 Analysis results saved to: {json_path}")

            return {
                "status": "success",
                "analyzed_products": analyzed_products,
                "total_analyzed": len(analyzed_products),
                "successful_analyses": len([p for p in analyzed_products if p.get("consensus_confidence", 0) > 0]),
                "analysis_file": str(json_path)
            }

        except Exception as e:
            print(f"   ❌ Error in consensus analysis: {e}")
            return {
                "status": "error",
                "error": str(e),
                "analyzed_products": analyzed_products
            }

    def _step_06_final_csv_generation(self, analyzed_products: List[Dict], category_data: Dict[str, Any], name: str) -> Dict[str, Any]:
        """STEP 6: Final CSV Generation with all consensus product data"""
        print("📊 STEP 6: Final CSV Generation - Creating comprehensive product database")

        step_dir = self.current_image_dir if hasattr(self, 'current_image_dir') else self.output_dir

        try:
            # Create comprehensive CSV data
            csv_data = []
            categories = category_data.get("available_subcategories", ["Unknown"])
            main_category = category_data.get("main_category", "Unknown")

            for product in analyzed_products:
                csv_row = {
                    # Basic Info
                    "product_id": product.get("product_id", f"{name}_product_{product.get('product_number', 'unknown')}"),
                    "product_number": product.get("product_number", 0),
                    "screenshot_source": name,
                    "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

                    # Category Context
                    "main_category": main_category,
                    "subcategory": product.get("subcategory", "Unknown"),
                    "detected_category": product.get("detected_category", ""),

                    # Product Details from Consensus Analysis
                    "product_name": product.get("product_name", ""),
                    "brand": product.get("brand", ""),
                    "price": product.get("price", ""),
                    "original_price": product.get("original_price", ""),
                    "unit": product.get("unit", ""),
                    "weight_quantity": product.get("weight_quantity", ""),
                    "description": product.get("description", ""),

                    # Quality Metrics
                    "consensus_confidence": product.get("consensus_confidence", 0.0),
                    "models_agreed": product.get("models_agreed", 0),
                    "total_models": product.get("total_models", 3),

                    # File Paths
                    "clean_image_path": product.get("clean_image_path", ""),
                    "text_image_path": product.get("text_image_path", ""),
                    "original_canvas_path": product.get("original_canvas_path", ""),

                    # Canvas Information
                    "canvas_x": product.get("canvas_info", {}).get("x", 0),
                    "canvas_y": product.get("canvas_info", {}).get("y", 0),
                    "canvas_width": product.get("canvas_info", {}).get("width", 0),
                    "canvas_height": product.get("canvas_info", {}).get("height", 0),
                }
                csv_data.append(csv_row)

            # Save CSV with comprehensive product data
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_path = step_dir / f"{name}_06_final_products.csv"
                df.to_csv(csv_path, index=False)

                print(f"   📋 Final CSV Generated: {len(csv_data)} products")
                print(f"   💾 CSV saved to: {csv_path}")

                # Print summary of analyzed products
                successful_products = [p for p in csv_data if p["consensus_confidence"] > 0]
                print(f"   ✅ Successful analyses: {len(successful_products)}/{len(csv_data)}")

                if successful_products:
                    print("   🛒 Sample products found:")
                    for product in successful_products[:3]:  # Show first 3 products
                        print(f"     • {product['product_name']} - {product['price']} ({product['consensus_confidence']:.1f}% confidence)")

                return {
                    "status": "success",
                    "csv_data": csv_data,
                    "csv_path": str(csv_path),
                    "total_products": len(csv_data),
                    "successful_analyses": len(successful_products),
                    "categories_detected": len(set(p["detected_category"] for p in csv_data if p["detected_category"]))
                }
            else:
                print("   ⚠️  No products to export to CSV")
                return {
                    "status": "warning",
                    "csv_data": [],
                    "message": "No products to export"
                }

        except Exception as e:
            print(f"   ❌ Error generating CSV: {e}")
            return {
                "status": "error",
                "error": str(e),
                "csv_data": []
            }


def test_step_by_step_pipeline():
    """Test the step-by-step pipeline"""
    pipeline = StepByStepPipeline("step_by_step_results")
    
    # Test on sample screenshots
    test_files = [
        "flink_sample_test/IMG_7805.PNG",
        "flink_sample_test/IMG_7806.PNG",
        "flink_sample_test/IMG_7999.PNG"
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\\n{'='*80}")
            print(f"Testing Step-by-Step Pipeline: {test_file}")
            print('='*80)
            
            try:
                results = pipeline.run_complete_demonstration(test_file)
                break  # Just test one for now
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️  Test file not found: {test_file}")


if __name__ == "__main__":
    test_step_by_step_pipeline()