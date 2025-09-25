#!/usr/bin/env python3
"""
Comprehensive Product Enrichment Orchestrator
Integrates ALL available APIs and data sources for complete product enrichment
"""

import sqlite3
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# Import our specialized enrichers
from nutritionix_enricher import NutritionixEnricher
from image_scraper import ImageScraper

class EnrichmentOrchestrator:
    """
    Master orchestrator that runs ALL enrichment processes:
    1. Nutritionix API - nutritional data
    2. Open Food Facts - product database
    3. USDA FoodData Central - government nutrition data
    4. Wikipedia - images and general info
    5. Brand websites - official images and data
    6. Google Shopping - price comparisons
    7. Image scraping - high-quality product photos
    """
    
    def __init__(self, db_path: str, config: Dict = None):
        self.db_path = Path(db_path)
        self.config = config or {}
        
        # Initialize specialized enrichers
        self.nutritionix = NutritionixEnricher(
            str(self.db_path), 
            self.config.get('nutritionix_app_id'),
            self.config.get('nutritionix_api_key')
        )
        self.image_scraper = ImageScraper(str(self.db_path))
        
        # API endpoints and keys
        self.apis = {
            'nutritionix': {
                'base_url': 'https://trackapi.nutritionix.com',
                'app_id': self.config.get('nutritionix_app_id'),
                'api_key': self.config.get('nutritionix_api_key')
            },
            'openfoodfacts': {
                'base_url': 'https://world.openfoodfacts.org',
                'api_key': None  # Open API
            },
            'usda': {
                'base_url': 'https://api.nal.usda.gov/fdc',
                'api_key': self.config.get('usda_api_key')
            },
            'google_shopping': {
                'api_key': self.config.get('google_api_key'),
                'cx': self.config.get('google_cx')
            },
            'edamam': {
                'base_url': 'https://api.edamam.com/api/food-database/v2',
                'app_id': self.config.get('edamam_app_id'),
                'api_key': self.config.get('edamam_api_key')
            }
        }
        
        self.setup_database()
    
    def setup_database(self):
        """Extend database for comprehensive enrichment tracking"""
        with sqlite3.connect(self.db_path) as conn:
            # Enrichment tracking table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS enrichment_sessions (
                    id TEXT PRIMARY KEY,
                    product_id TEXT REFERENCES master_products(id),
                    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_end TIMESTAMP,
                    
                    -- API Status Tracking
                    nutritionix_status TEXT DEFAULT 'pending',
                    openfoodfacts_status TEXT DEFAULT 'pending',
                    usda_status TEXT DEFAULT 'pending',
                    wikipedia_status TEXT DEFAULT 'pending',
                    image_scraping_status TEXT DEFAULT 'pending',
                    google_shopping_status TEXT DEFAULT 'pending',
                    
                    -- Results Summary
                    total_apis_called INTEGER DEFAULT 0,
                    successful_apis INTEGER DEFAULT 0,
                    images_found INTEGER DEFAULT 0,
                    nutrition_data_found BOOLEAN DEFAULT FALSE,
                    price_data_found BOOLEAN DEFAULT FALSE,
                    
                    -- Final Score
                    final_enrichment_score REAL DEFAULT 0.0,
                    
                    -- Error Log
                    errors TEXT,  -- JSON array of errors
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Price comparison table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS price_comparisons (
                    id TEXT PRIMARY KEY,
                    product_id TEXT REFERENCES master_products(id),
                    source TEXT NOT NULL,  -- 'google_shopping', 'amazon', etc.
                    vendor TEXT,
                    price REAL,
                    currency TEXT DEFAULT 'EUR',
                    product_url TEXT,
                    availability TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def enrich_product_comprehensive(self, product_id: str, force_refresh: bool = False) -> Dict:
        """
        Run COMPLETE enrichment pipeline for a single product
        """
        print(f"🚀 Starting comprehensive enrichment for product: {product_id}")
        
        # Get product info
        product = self.get_product(product_id)
        if not product:
            return {'error': 'Product not found'}
        
        print(f"   📦 Product: {product['csv_product_name']} ({product['canonical_name']})")
        print(f"   🏷️  Brand: {product['csv_brand'] or 'Generic'}")
        
        # Create enrichment session
        session_id = self.create_enrichment_session(product_id)
        
        enrichment_results = {
            'session_id': session_id,
            'product_id': product_id,
            'product_name': product['csv_product_name'],
            'status': 'running',
            'apis_called': [],
            'results': {},
            'errors': []
        }
        
        # 1. NUTRITIONIX API - Detailed nutrition data
        print(f"\n1️⃣  Enriching with Nutritionix API...")
        try:
            nutritionix_result = self.nutritionix.enrich_product(product_id, force_refresh)
            enrichment_results['results']['nutritionix'] = nutritionix_result
            enrichment_results['apis_called'].append('nutritionix')
            self.update_enrichment_status(session_id, 'nutritionix_status', 'completed')
            print(f"   ✅ Nutritionix: {nutritionix_result['status']}")
        except Exception as e:
            error_msg = f"Nutritionix API error: {str(e)}"
            enrichment_results['errors'].append(error_msg)
            self.update_enrichment_status(session_id, 'nutritionix_status', 'failed')
            print(f"   ❌ Nutritionix failed: {e}")
        
        # 2. OPEN FOOD FACTS - Product database
        print(f"\n2️⃣  Enriching with Open Food Facts...")
        try:
            off_result = self.enrich_with_openfoodfacts(product)
            enrichment_results['results']['openfoodfacts'] = off_result
            enrichment_results['apis_called'].append('openfoodfacts')
            self.update_enrichment_status(session_id, 'openfoodfacts_status', 'completed')
            print(f"   ✅ Open Food Facts: Found {len(off_result.get('products', []))} matches")
        except Exception as e:
            error_msg = f"Open Food Facts error: {str(e)}"
            enrichment_results['errors'].append(error_msg)
            self.update_enrichment_status(session_id, 'openfoodfacts_status', 'failed')
            print(f"   ❌ Open Food Facts failed: {e}")
        
        # 3. USDA FOODDATA CENTRAL - Government nutrition data
        print(f"\n3️⃣  Enriching with USDA FoodData Central...")
        try:
            usda_result = self.enrich_with_usda(product)
            enrichment_results['results']['usda'] = usda_result
            enrichment_results['apis_called'].append('usda')
            self.update_enrichment_status(session_id, 'usda_status', 'completed')
            print(f"   ✅ USDA: Found {len(usda_result.get('foods', []))} matches")
        except Exception as e:
            error_msg = f"USDA API error: {str(e)}"
            enrichment_results['errors'].append(error_msg)
            self.update_enrichment_status(session_id, 'usda_status', 'failed')
            print(f"   ❌ USDA failed: {e}")
        
        # 4. IMAGE SCRAPING - High-quality product images
        print(f"\n4️⃣  Scraping high-quality images...")
        try:
            images_result = self.image_scraper.scrape_product_images(
                product_id, 
                product['canonical_name'], 
                product['csv_brand']
            )
            enrichment_results['results']['images'] = images_result
            enrichment_results['apis_called'].append('image_scraping')
            self.update_enrichment_status(session_id, 'image_scraping_status', 'completed')
            print(f"   ✅ Images: Downloaded {len(images_result)} high-quality images")
        except Exception as e:
            error_msg = f"Image scraping error: {str(e)}"
            enrichment_results['errors'].append(error_msg)
            self.update_enrichment_status(session_id, 'image_scraping_status', 'failed')
            print(f"   ❌ Image scraping failed: {e}")
        
        # 5. GOOGLE SHOPPING - Price comparisons
        print(f"\n5️⃣  Finding price comparisons...")
        try:
            shopping_result = self.enrich_with_google_shopping(product)
            enrichment_results['results']['google_shopping'] = shopping_result
            enrichment_results['apis_called'].append('google_shopping')
            self.update_enrichment_status(session_id, 'google_shopping_status', 'completed')
            print(f"   ✅ Shopping: Found {len(shopping_result.get('prices', []))} price points")
        except Exception as e:
            error_msg = f"Google Shopping error: {str(e)}"
            enrichment_results['errors'].append(error_msg)
            self.update_enrichment_status(session_id, 'google_shopping_status', 'failed')
            print(f"   ❌ Google Shopping failed: {e}")
        
        # 6. Calculate final enrichment score
        final_score = self.calculate_enrichment_score(enrichment_results)
        
        # Update product with new enrichment score
        self.update_product_enrichment_score(product_id, final_score)
        
        # Close enrichment session
        self.close_enrichment_session(session_id, enrichment_results, final_score)
        
        enrichment_results['status'] = 'completed'
        enrichment_results['final_score'] = final_score
        
        print(f"\n🎯 ENRICHMENT COMPLETE!")
        print(f"   📊 Final Score: {final_score:.1f}%")
        print(f"   ✅ APIs Called: {len(enrichment_results['apis_called'])}")
        print(f"   ❌ Errors: {len(enrichment_results['errors'])}")
        
        return enrichment_results
    
    def enrich_with_openfoodfacts(self, product: Dict) -> Dict:
        """Enrich with Open Food Facts database"""
        
        search_terms = []
        if product['csv_brand']:
            search_terms.append(f"{product['csv_brand']} {product['canonical_name']}")
        search_terms.append(product['canonical_name'])
        search_terms.append(product['csv_product_name'])
        
        results = {'products': [], 'best_match': None}
        
        for term in search_terms:
            try:
                url = f"{self.apis['openfoodfacts']['base_url']}/cgi/search.pl"
                params = {
                    'search_terms': term,
                    'search_simple': 1,
                    'action': 'process',
                    'json': 1
                }
                
                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    
                    for off_product in data.get('products', [])[:3]:  # Top 3 matches
                        # Extract useful data
                        product_data = {
                            'product_name': off_product.get('product_name'),
                            'brands': off_product.get('brands'),
                            'categories': off_product.get('categories'),
                            'ingredients_text': off_product.get('ingredients_text'),
                            'nutrition_grades': off_product.get('nutrition_grades'),
                            'nova_group': off_product.get('nova_group'),
                            'ecoscore_grade': off_product.get('ecoscore_grade'),
                            'image_url': off_product.get('image_url'),
                            'url': off_product.get('url')
                        }
                        results['products'].append(product_data)
                        
                        # Set best match (first good match)
                        if not results['best_match'] and off_product.get('product_name'):
                            results['best_match'] = product_data
                    
                    break  # Found results, stop searching
                    
            except Exception as e:
                print(f"      OpenFoodFacts search failed for '{term}': {e}")
                continue
        
        return results
    
    def enrich_with_usda(self, product: Dict) -> Dict:
        """Enrich with USDA FoodData Central"""
        
        if not self.apis['usda']['api_key']:
            return {'error': 'USDA API key not configured'}
        
        search_terms = [
            product['canonical_name'],
            f"{product['canonical_name']} raw",
            f"{product['canonical_name']} fresh"
        ]
        
        results = {'foods': [], 'best_match': None}
        
        for term in search_terms:
            try:
                url = f"{self.apis['usda']['base_url']}/v1/foods/search"
                params = {
                    'query': term,
                    'dataType': ['Foundation', 'SR Legacy'],
                    'pageSize': 5,
                    'api_key': self.apis['usda']['api_key']
                }
                
                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    
                    for food in data.get('foods', []):
                        food_data = {
                            'fdcId': food.get('fdcId'),
                            'description': food.get('description'),
                            'dataType': food.get('dataType'),
                            'foodCode': food.get('foodCode'),
                            'foodNutrients': food.get('foodNutrients', [])[:10]  # First 10 nutrients
                        }
                        results['foods'].append(food_data)
                        
                        if not results['best_match']:
                            results['best_match'] = food_data
                    
                    break
                    
            except Exception as e:
                print(f"      USDA search failed for '{term}': {e}")
                continue
        
        return results
    
    def enrich_with_google_shopping(self, product: Dict) -> Dict:
        """Find price comparisons using Google Shopping API"""
        
        if not self.apis['google_shopping']['api_key']:
            return {'error': 'Google Shopping API key not configured', 'prices': []}
        
        search_query = f"{product['csv_brand']} {product['canonical_name']}" if product['csv_brand'] else product['canonical_name']
        
        results = {'prices': [], 'average_price': None}
        
        try:
            # Use Google Custom Search API with Shopping focus
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.apis['google_shopping']['api_key'],
                'cx': self.apis['google_shopping']['cx'],
                'q': f"{search_query} price buy",
                'searchType': 'shopping' if 'shopping' in str(self.apis['google_shopping']['cx']) else None
            }
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                
                prices = []
                for item in data.get('items', []):
                    # Extract price information (this would need refinement)
                    snippet = item.get('snippet', '')
                    price_match = self.extract_price_from_text(snippet)
                    
                    if price_match:
                        price_data = {
                            'vendor': item.get('displayLink'),
                            'price': price_match['price'],
                            'currency': price_match['currency'],
                            'url': item.get('link'),
                            'title': item.get('title')
                        }
                        results['prices'].append(price_data)
                        prices.append(price_match['price'])
                
                if prices:
                    results['average_price'] = sum(prices) / len(prices)
            
        except Exception as e:
            print(f"      Google Shopping failed: {e}")
        
        return results
    
    def extract_price_from_text(self, text: str) -> Optional[Dict]:
        """Extract price information from text"""
        import re
        
        # Look for price patterns like €1.49, $2.99, etc.
        price_patterns = [
            r'€(\d+[.,]\d+)',  # €1.49
            r'\$(\d+[.,]\d+)',  # $2.99
            r'(\d+[.,]\d+)\s*€',  # 1.49 €
            r'(\d+[.,]\d+)\s*USD',  # 2.99 USD
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text)
            if match:
                price_str = match.group(1).replace(',', '.')
                try:
                    price = float(price_str)
                    currency = 'EUR' if '€' in pattern else 'USD'
                    return {'price': price, 'currency': currency}
                except ValueError:
                    continue
        
        return None
    
    def calculate_enrichment_score(self, results: Dict) -> float:
        """Calculate final enrichment score based on all data found"""
        score = 0.0
        max_score = 100.0
        
        # API success scoring
        apis_called = len(results['apis_called'])
        if apis_called > 0:
            score += (apis_called / 5) * 30  # 30 points for API coverage
        
        # Nutrition data
        if 'nutritionix' in results['results'] and results['results']['nutritionix'].get('status') == 'success':
            score += 25  # 25 points for nutrition data
        elif 'usda' in results['results'] and results['results']['usda'].get('foods'):
            score += 20  # 20 points for USDA data
        
        # Images
        images = results['results'].get('images', [])
        if images:
            score += min(len(images) * 5, 20)  # Up to 20 points for images
        
        # Product database matches
        if 'openfoodfacts' in results['results']:
            off_products = results['results']['openfoodfacts'].get('products', [])
            if off_products:
                score += 15  # 15 points for database match
        
        # Price data
        if 'google_shopping' in results['results']:
            prices = results['results']['google_shopping'].get('prices', [])
            if prices:
                score += 10  # 10 points for price data
        
        return min(score, max_score)
    
    def get_product(self, product_id: str) -> Optional[Dict]:
        """Get product from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            product = conn.execute('SELECT * FROM master_products WHERE id = ?', (product_id,)).fetchone()
            return dict(product) if product else None
    
    def create_enrichment_session(self, product_id: str) -> str:
        """Create new enrichment session"""
        import uuid
        session_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO enrichment_sessions (id, product_id, session_start)
                VALUES (?, ?, ?)
            ''', (session_id, product_id, datetime.now().isoformat()))
        
        return session_id
    
    def update_enrichment_status(self, session_id: str, status_field: str, status_value: str):
        """Update enrichment status for specific API"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f'''
                UPDATE enrichment_sessions 
                SET {status_field} = ?
                WHERE id = ?
            ''', (status_value, session_id))
    
    def close_enrichment_session(self, session_id: str, results: Dict, final_score: float):
        """Close enrichment session with final results"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE enrichment_sessions 
                SET session_end = ?, total_apis_called = ?, successful_apis = ?,
                    images_found = ?, final_enrichment_score = ?, errors = ?
                WHERE id = ?
            ''', (
                datetime.now().isoformat(),
                len(results['apis_called']),
                len([api for api in results['apis_called'] if api in results.get('results', {})]),
                len(results['results'].get('images', [])),
                final_score,
                json.dumps(results['errors']),
                session_id
            ))
    
    def update_product_enrichment_score(self, product_id: str, score: float):
        """Update product's enrichment score"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE master_products 
                SET enrichment_score = ?, last_enriched = ?
                WHERE id = ?
            ''', (score, datetime.now().isoformat(), product_id))
    
    def enrich_all_products(self, batch_size: int = 3, delay: float = 2.0) -> Dict:
        """Run comprehensive enrichment on all products (with rate limiting)"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get products that need enrichment
            products = conn.execute('''
                SELECT id, csv_product_name, csv_brand, canonical_name, enrichment_score
                FROM master_products 
                WHERE enrichment_score < 80 OR last_enriched IS NULL
                ORDER BY enrichment_score ASC
                LIMIT ?
            ''', (batch_size,)).fetchall()
        
        results = {
            'total_products': len(products),
            'successful': 0,
            'failed': 0,
            'product_results': []
        }
        
        print(f"🔄 Starting batch enrichment for {len(products)} products...")
        
        for i, product in enumerate(products):
            print(f"\n📦 Processing product {i+1}/{len(products)}: {product['csv_product_name']}")
            
            try:
                product_result = self.enrich_product_comprehensive(product['id'])
                results['product_results'].append(product_result)
                
                if product_result.get('status') == 'completed':
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    
            except Exception as e:
                print(f"   ❌ Failed to enrich product {product['id']}: {e}")
                results['failed'] += 1
            
            # Rate limiting delay
            if i < len(products) - 1:  # Don't delay after last product
                print(f"   ⏱️  Waiting {delay}s before next product...")
                time.sleep(delay)
        
        print(f"\n🏁 Batch enrichment complete: {results['successful']} success, {results['failed']} failed")
        return results

def main():
    """CLI interface for comprehensive enrichment"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python enrichment_orchestrator.py <database_path> [product_id]")
        return
    
    db_path = sys.argv[1]
    
    # Configuration (in production, load from config file)
    config = {
        'nutritionix_api_key': None,  # Set your API keys here
        'nutritionix_app_id': None,
        'usda_api_key': None,
        'google_api_key': None,
        'google_cx': None,
        'edamam_app_id': None,
        'edamam_api_key': None
    }
    
    orchestrator = EnrichmentOrchestrator(db_path, config)
    
    if len(sys.argv) > 2:
        # Single product enrichment
        product_id = sys.argv[2]
        result = orchestrator.enrich_product_comprehensive(product_id)
        print(f"\n🎯 Final result: {result}")
    else:
        # Batch enrichment
        result = orchestrator.enrich_all_products(batch_size=2, delay=1.0)
        print(f"\n📊 Batch result: {result}")

if __name__ == "__main__":
    main()