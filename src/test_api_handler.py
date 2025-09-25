#!/usr/bin/env python3
"""
Test API Handler for Different Input Scenarios
Handles image-only, text-only, text+image, and new product inputs
"""

import os
import uuid
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from werkzeug.utils import secure_filename
from PIL import Image
import json

class TestAPIHandler:
    def __init__(self, db_path: str, upload_dir: str = "uploads"):
        self.db_path = Path(db_path)
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        
    def handle_image_only(self, image_file) -> Dict[str, Any]:
        """
        Test Scenario 1: Image Only
        - Save uploaded image
        - Use vision analysis to identify food
        - Create product entry
        - Enrich with images and nutrition
        """
        try:
            # Save uploaded image
            filename = secure_filename(image_file.filename)
            file_path = self.upload_dir / f"{uuid.uuid4()}_{filename}"
            image_file.save(str(file_path))
            
            # Real vision analysis using Claude/Gemini
            print(f"🔍 Starting real vision analysis for: {file_path}")
            identified_food = self.mock_vision_analysis(str(file_path))
            print(f"✅ Vision analysis complete: {identified_food.get('product_name', 'Unknown')}")
            
            # Check if non-food item detected
            if not identified_food.get('is_food', True):
                return {
                    'status': 'error',
                    'error': f'Non-food items detected: {", ".join(identified_food.get("items_detected", ["Unknown items"]))}',
                    'error_type': 'non_food',
                    'items_detected': identified_food.get('items_detected', []),
                    'message': 'Please upload an image containing food items only'
                }
            
            # Create product from vision analysis
            product_data = {
                'name': identified_food['product_name'],
                'brand': identified_food.get('brand'),
                'category': identified_food.get('category', 'food'),
                'confidence': identified_food['confidence'],
                'comprehensive_analysis': identified_food.get('comprehensive_analysis'),
                'variety': identified_food.get('variety'),
                'ripeness': identified_food.get('ripeness'),
                'size_estimate': identified_food.get('size_estimate'),
                'certifications': identified_food.get('certifications', []),
                'estimated_retail_value': identified_food.get('estimated_retail_value'),
                'plu_code': identified_food.get('plu_code'),
                'packaging_type': identified_food.get('packaging_type')
            }
            
            # Create product entry
            product_id = self.create_product_entry(product_data)
            
            # Store original uploaded image as PRIMARY image
            uploaded_image_id = self.store_original_image(product_id, str(file_path), image_file.filename, is_primary=True)
            
            # Enrich with additional images and nutrition  
            enrichment_result = self.enrich_product_comprehensive(product_id)
            
            # Add uploaded image info to result
            enrichment_result['uploaded_image_path'] = str(file_path)
            enrichment_result['uploaded_image_id'] = uploaded_image_id
            
            return {
                'status': 'success',
                'product_id': product_id,
                'identified_food': identified_food['product_name'],
                'confidence': identified_food['confidence'],
                'images_found': enrichment_result['images_found'],
                'nutrition_found': enrichment_result['nutrition_found'],
                'enrichment_score': enrichment_result['enrichment_score']
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Image analysis failed: {str(e)}'
            }
    
    def handle_text_only(self, description: str) -> Dict[str, Any]:
        """
        Test Scenario 2: Text Only
        - Parse product name and brand from text
        - Create product entry
        - Enrich with images and nutrition
        """
        try:
            # Parse text to extract product info
            parsed_info = self.parse_product_description(description)
            
            # Create product entry
            product_id = self.create_product_entry(parsed_info)
            
            # Enrich with images and nutrition
            enrichment_result = self.enrich_product_comprehensive(product_id)
            
            return {
                'status': 'success',
                'product_id': product_id,
                'parsed_name': parsed_info['name'],
                'parsed_brand': parsed_info.get('brand'),
                'images_found': enrichment_result['images_found'],
                'nutrition_found': enrichment_result['nutrition_found'],
                'enrichment_score': enrichment_result['enrichment_score']
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Text analysis failed: {str(e)}'
            }
    
    def handle_text_plus_image(self, product_name: str, image_file) -> Dict[str, Any]:
        """
        Test Scenario 3: Text + Image
        - Analyze both text and image
        - Cross-validate for consistency
        - Create optimized product entry
        - Enrich comprehensively
        """
        try:
            results = {}
            
            # Handle text analysis
            if product_name:
                text_analysis = self.parse_product_description(product_name)
                results.update(text_analysis)
            
            # Handle image analysis
            if image_file:
                filename = secure_filename(image_file.filename)
                file_path = self.upload_dir / f"{uuid.uuid4()}_{filename}"
                image_file.save(str(file_path))
                
                vision_analysis = self.mock_vision_analysis(str(file_path))
                
                # Cross-validate text vs image
                if product_name:
                    match_confidence = self.calculate_match_confidence(text_analysis, vision_analysis)
                    results['match_confidence'] = match_confidence
                    
                    # Use the more confident result
                    if match_confidence > 70:
                        results['final_name'] = text_analysis['name']
                        results['final_brand'] = text_analysis.get('brand', vision_analysis.get('brand'))
                        results['ai_brand'] = vision_analysis.get('brand')
                        results['ai_category'] = vision_analysis.get('category')
                        results['ai_confidence'] = vision_analysis.get('confidence', 70) / 100.0
                    else:
                        results['final_name'] = vision_analysis['product_name']
                        results['final_brand'] = vision_analysis.get('brand')
                        results['ai_brand'] = vision_analysis.get('brand')
                        results['ai_category'] = vision_analysis.get('category')
                        results['ai_confidence'] = vision_analysis.get('confidence', 70) / 100.0
                else:
                    # Only image provided
                    results['final_name'] = vision_analysis['product_name']
                    results['final_brand'] = vision_analysis.get('brand')
                    results['ai_brand'] = vision_analysis.get('brand')
                    results['ai_category'] = vision_analysis.get('category')
                    results['ai_confidence'] = vision_analysis.get('confidence', 70) / 100.0
                    results['match_confidence'] = vision_analysis['confidence']
                
                # Store original image
                self.store_original_image(results.get('product_id', str(uuid.uuid4())), str(file_path), image_file.filename, is_primary=True)
            
            # Create product entry
            product_data = {
                'name': results.get('final_name', results.get('name')),
                'brand': results.get('final_brand', results.get('brand')),
                'category': results.get('category', 'food'),
                'ai_brand': results.get('ai_brand'),
                'ai_category': results.get('ai_category'),
                'ai_confidence': results.get('ai_confidence')
            }
            product_id = self.create_product_entry(product_data)
            results['product_id'] = product_id
            
            # Enrich comprehensively
            enrichment_result = self.enrich_product_comprehensive(product_id)
            
            return {
                'status': 'success',
                **results,
                'images_found': enrichment_result['images_found'],
                'nutrition_found': enrichment_result['nutrition_found'],
                'enrichment_score': enrichment_result['enrichment_score']
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Combined analysis failed: {str(e)}'
            }
    
    def handle_new_product(self, name: str, brand: str = None, price: str = None, 
                          weight: str = None, image_file = None) -> Dict[str, Any]:
        """
        Test Scenario 4: New Product Entry
        - Create product with all provided info
        - Normalize and canonicalize data
        - Enrich with additional data
        """
        try:
            # Create comprehensive product data
            product_data = {
                'name': name,
                'brand': brand,
                'price': price,
                'weight': weight,
                'category': self.infer_category(name)
            }
            
            # Create product entry
            product_id = self.create_product_entry(product_data)
            
            # Handle uploaded image if provided
            if image_file:
                filename = secure_filename(image_file.filename)
                file_path = self.upload_dir / f"{uuid.uuid4()}_{filename}"
                image_file.save(str(file_path))
                self.store_original_image(product_id, str(file_path), image_file.filename)
            
            # Enrich comprehensively
            enrichment_result = self.enrich_product_comprehensive(product_id)
            
            # Get final product data
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                product = conn.execute('SELECT * FROM master_products WHERE id = ?', (product_id,)).fetchone()
            
            return {
                'status': 'success',
                'product_id': product_id,
                'canonical_name': product['canonical_name'],
                'canonical_brand': product['canonical_brand'],
                'category': product['canonical_category'],
                'normalized_price': product['normalized_price_eur'],
                'images_found': enrichment_result['images_found'],
                'nutrition_found': enrichment_result['nutrition_found'],
                'enrichment_score': enrichment_result['enrichment_score']
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Product creation failed: {str(e)}'
            }
    
    def mock_vision_analysis(self, image_path: str) -> Dict[str, Any]:
        """Real vision analysis using Claude or Gemini APIs"""
        
        # Use the real vision analyzer with explicit API key
        from real_vision_analyzer import RealVisionAnalyzer
        import os
        import sys
        
        # Add parent directory to path for config import
        sys.path.append('..')
        try:
            from vision_api_config import get_vision_api_key
            api_key, model = get_vision_api_key()
            print(f"🔑 API Key found: {bool(api_key)}, Model: {model}")
        except ImportError:
            api_key, model = None, None
            print("⚠️ Could not import vision_api_config")
        
        analyzer = RealVisionAnalyzer(api_key, model)
        
        # Get comprehensive real analysis
        analysis_result = analyzer.analyze_food_image(image_path)
        
        # Check if it's non-food
        if not analysis_result.get('is_food', True):
            return {
                'product_name': 'Non-Food Item',
                'variety': None,
                'brand': None,
                'category': 'non-food',
                'confidence': int(analysis_result.get('overall_confidence', 0.95) * 100),
                'is_food': False,
                'items_detected': analysis_result.get('items_detected', []),
                'comprehensive_analysis': analysis_result,
                'error_message': 'This image does not contain food items'
            }
        
        # Extract primary food item for compatibility
        primary_item = analysis_result['food_items'][0] if analysis_result['food_items'] else {}
        
        # Convert to expected format while preserving comprehensive data
        return {
            'product_name': primary_item.get('name', 'Unknown Food'),
            'variety': primary_item.get('variety'),
            'brand': analysis_result.get('brand'),
            'category': analysis_result.get('recommended_database_category', 'food'),
            'confidence': int(analysis_result.get('overall_confidence', 0.7) * 100),
            'is_food': True,
            
            # Additional comprehensive data
            'comprehensive_analysis': analysis_result,
            'pricing_info': analysis_result.get('pricing_info', {}),
            'origin_info': analysis_result.get('origin_info', {}),
            'nutritional_clues': analysis_result.get('nutritional_visible', {}),
            'certifications': analysis_result.get('certifications', []),
            'all_text_extracted': analysis_result.get('all_text_extracted', []),
            'estimated_retail_value': analysis_result.get('estimated_retail_value'),
            'plu_code': analysis_result.get('pricing_info', {}).get('plu_code'),
            'packaging_type': analysis_result.get('packaging_type'),
            'ripeness': primary_item.get('ripeness'),
            'size_estimate': primary_item.get('size_estimate'),
            'quantity_count': primary_item.get('quantity_count', 1)
        }
    
    def parse_product_description(self, description: str) -> Dict[str, Any]:
        """Parse product description to extract name, brand, etc."""
        desc_lower = description.lower()
        
        # Common brand patterns
        brands = ['chiquita', 'bio', 'organic', 'pink lady', 'driscoll', 'dole', 'del monte']
        found_brand = None
        
        for brand in brands:
            if brand in desc_lower:
                found_brand = brand.title()
                break
        
        # Remove brand from product name if found
        clean_name = description
        if found_brand:
            clean_name = description.replace(found_brand.lower(), '').replace(found_brand, '').strip()
        
        # Infer category
        category = self.infer_category(clean_name)
        
        return {
            'name': clean_name or description,
            'brand': found_brand,
            'category': category
        }
    
    def infer_category(self, name: str) -> str:
        """Infer food category from product name"""
        name_lower = name.lower()
        
        if any(fruit in name_lower for fruit in ['apple', 'banana', 'orange', 'grape', 'berry', 'avocado']):
            return 'fruit'
        elif any(veg in name_lower for veg in ['carrot', 'lettuce', 'spinach', 'tomato', 'potato']):
            return 'vegetable'
        elif any(dairy in name_lower for dairy in ['milk', 'cheese', 'yogurt', 'butter']):
            return 'dairy'
        elif any(meat in name_lower for meat in ['chicken', 'beef', 'pork', 'fish']):
            return 'protein'
        else:
            return 'food'
    
    def calculate_match_confidence(self, text_analysis: Dict, vision_analysis: Dict) -> int:
        """Calculate how well text and vision analysis match"""
        text_name = text_analysis['name'].lower()
        vision_name = vision_analysis['product_name'].lower()
        
        # Simple word overlap calculation
        text_words = set(text_name.split())
        vision_words = set(vision_name.split())
        
        if text_words & vision_words:
            overlap = len(text_words & vision_words)
            total = len(text_words | vision_words)
            return int((overlap / total) * 100)
        else:
            return 50  # Neutral confidence
    
    def create_product_entry(self, product_data: Dict) -> str:
        """Create a new product entry in the database with comprehensive data"""
        product_id = str(uuid.uuid4())
        
        # Create product signature for deduplication
        signature_text = f"{product_data.get('brand', 'generic')}_{product_data['name']}_{product_data.get('weight', '')}"
        signature = hashlib.md5(signature_text.encode()).hexdigest()[:16]
        
        # Normalize price from various sources
        price_eur = None
        if product_data.get('price'):
            price_str = product_data['price'].replace('€', '').replace('$', '').replace(',', '.').strip()
            try:
                price_eur = float(price_str)
            except:
                pass
        
        # Try to extract price from comprehensive vision analysis
        if not price_eur and product_data.get('estimated_retail_value'):
            try:
                price_str = product_data['estimated_retail_value'].replace('€', '').replace('$', '').split()[0]
                price_eur = float(price_str)
            except:
                pass
        
        # Build comprehensive metadata from vision analysis
        vision_metadata = {}
        if product_data.get('comprehensive_analysis'):
            analysis = product_data['comprehensive_analysis']
            vision_metadata = {
                'variety': product_data.get('variety'),
                'ripeness': product_data.get('ripeness'),
                'size_estimate': product_data.get('size_estimate'),
                'quantity_count': product_data.get('quantity_count'),
                'certifications': product_data.get('certifications', []),
                'packaging_type': product_data.get('packaging_type'),
                'plu_code': product_data.get('plu_code'),
                'origin_country': analysis.get('origin_info', {}).get('country_of_origin'),
                'growing_method': analysis.get('origin_info', {}).get('growing_method'),
                'nutritional_claims': analysis.get('nutritional_visible', {}).get('health_claims', []),
                'all_text_extracted': product_data.get('all_text_extracted', []),
                'analysis_confidence': product_data.get('confidence'),
                'estimated_value': product_data.get('estimated_retail_value')
            }
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO master_products (
                    id, product_signature, csv_brand, csv_product_name, csv_price, csv_weight,
                    canonical_name, canonical_brand, canonical_category, 
                    ai_brand, ai_category, ai_confidence,
                    normalized_price_eur, enrichment_score, vision_metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                product_id, signature, product_data.get('brand'), product_data['name'],
                product_data.get('price'), product_data.get('weight'),
                product_data['name'], product_data.get('brand'), product_data.get('category'),
                product_data.get('ai_brand'), product_data.get('ai_category'), product_data.get('ai_confidence'),
                price_eur, 25.0, json.dumps(vision_metadata), datetime.now().isoformat()
            ))
        
        return product_id
    
    def store_original_image(self, product_id: str, file_path: str, original_name: str, is_primary: bool = False):
        """Store the original uploaded image"""
        try:
            from PIL import Image
            
            # Get image dimensions
            with Image.open(file_path) as img:
                width, height = img.size
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            image_id = str(uuid.uuid4())
            
            with sqlite3.connect(self.db_path) as conn:
                # If this is primary, unset other primary images for this product
                if is_primary:
                    conn.execute('UPDATE product_images SET is_primary = FALSE WHERE product_id = ?', (product_id,))
                
                conn.execute('''
                    INSERT INTO product_images (
                        id, product_id, image_path, original_filename, 
                        width, height, file_size, file_exists, is_primary, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    image_id, product_id, file_path, original_name,
                    width, height, file_size, True, is_primary, datetime.now().isoformat()
                ))
            
            print(f"✅ Stored uploaded image: {original_name} ({'PRIMARY' if is_primary else 'secondary'})")
            return image_id
            
        except Exception as e:
            print(f"Warning: Could not store original image: {e}")
            return None
    
    def enrich_product_comprehensive(self, product_id: str) -> Dict[str, Any]:
        """Enrich product with images and nutrition data"""
        try:
            # Get product info
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                product = conn.execute('SELECT * FROM master_products WHERE id = ?', (product_id,)).fetchone()
            
            if not product:
                return {'images_found': 0, 'nutrition_found': False, 'enrichment_score': 0}
            
            # Enrich with images using simple enricher
            from simple_image_enricher import SimpleImageEnricher
            enricher = SimpleImageEnricher(str(self.db_path))
            
            images = enricher.enrich_product_images(
                product_id, 
                product['csv_product_name'], 
                product['csv_brand']
            )
            
            # Update enrichment score
            base_score = 10  # Base score for having a product entry
            image_score = min(len(images) * 15, 60)  # Up to 60 points for images
            nutrition_score = 0  # TODO: Add nutrition API integration
            
            final_score = base_score + image_score + nutrition_score
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE master_products 
                    SET enrichment_score = ?, last_enriched = ?
                    WHERE id = ?
                ''', (final_score, datetime.now().isoformat(), product_id))
            
            return {
                'images_found': len(images),
                'nutrition_found': False,  # TODO: Implement nutrition API
                'enrichment_score': final_score
            }
            
        except Exception as e:
            print(f"Enrichment error: {e}")
            return {'images_found': 0, 'nutrition_found': False, 'enrichment_score': 10}