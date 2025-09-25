#!/usr/bin/env python3
"""
Nutritionix API Integration
Fetches detailed nutritional data for food products
"""

import requests
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import time

class NutritionixEnricher:
    """
    Enriches product database with nutritional information from Nutritionix API
    """
    
    def __init__(self, db_path: str, app_id: str = None, app_key: str = None):
        self.db_path = Path(db_path)
        self.app_id = app_id or "your_nutritionix_app_id"
        self.app_key = app_key or "your_nutritionix_app_key"
        self.base_url = "https://trackapi.nutritionix.com/v2"
        self.setup_database()
    
    def setup_database(self):
        """Extend database schema for nutritional data"""
        with sqlite3.connect(self.db_path) as conn:
            # Add nutritional data columns to master_products table
            try:
                conn.execute('''
                    ALTER TABLE master_products ADD COLUMN nutritionix_id TEXT;
                ''')
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                conn.execute('''
                    ALTER TABLE master_products ADD COLUMN nutrition_data TEXT;
                ''')
            except sqlite3.OperationalError:
                pass
            
            try:
                conn.execute('''
                    ALTER TABLE master_products ADD COLUMN last_enriched TIMESTAMP;
                ''')
            except sqlite3.OperationalError:
                pass
            
            # Create nutritional facts table for detailed storage
            conn.execute('''
                CREATE TABLE IF NOT EXISTS nutrition_facts (
                    id TEXT PRIMARY KEY,
                    product_id TEXT REFERENCES master_products(id),
                    source TEXT DEFAULT 'nutritionix',
                    
                    -- Serving Information
                    serving_size REAL,
                    serving_unit TEXT,
                    serving_weight_grams REAL,
                    
                    -- Macronutrients (per serving)
                    calories REAL,
                    total_fat REAL,
                    saturated_fat REAL,
                    trans_fat REAL,
                    cholesterol REAL,
                    sodium REAL,
                    total_carbs REAL,
                    dietary_fiber REAL,
                    sugars REAL,
                    added_sugars REAL,
                    protein REAL,
                    
                    -- Vitamins & Minerals
                    vitamin_a_iu REAL,
                    vitamin_c REAL,
                    vitamin_d REAL,
                    vitamin_e REAL,
                    vitamin_k REAL,
                    thiamin REAL,
                    riboflavin REAL,
                    niacin REAL,
                    vitamin_b6 REAL,
                    folate REAL,
                    vitamin_b12 REAL,
                    calcium REAL,
                    iron REAL,
                    magnesium REAL,
                    phosphorus REAL,
                    potassium REAL,
                    zinc REAL,
                    
                    -- Additional Information
                    tags TEXT,  -- JSON array of tags (organic, gluten-free, etc.)
                    ingredients TEXT,
                    allergens TEXT,
                    
                    -- Metadata
                    confidence_score REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create enrichment queue for batch processing
            conn.execute('''
                CREATE TABLE IF NOT EXISTS enrichment_queue (
                    id TEXT PRIMARY KEY,
                    product_id TEXT REFERENCES master_products(id),
                    search_query TEXT,
                    priority INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'pending',  -- pending, processing, completed, failed
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP
                )
            ''')
    
    def enrich_product(self, product_id: str, force_refresh: bool = False) -> Dict:
        """
        Enrich a single product with nutritional data
        """
        # Get product from database
        product = self.get_product_by_id(product_id)
        if not product:
            return {'error': 'Product not found'}
        
        # Check if already enriched recently (unless forcing refresh)
        if not force_refresh and product.get('last_enriched'):
            last_enriched = datetime.fromisoformat(product['last_enriched'])
            if (datetime.now() - last_enriched).days < 30:
                return {'status': 'already_enriched', 'data': json.loads(product.get('nutrition_data', '{}'))}
        
        # Create search query
        search_query = self.build_search_query(product)
        
        try:
            # Search for the food item
            nutrition_data = self.search_nutrition(search_query)
            
            if not nutrition_data:
                return {'error': 'No nutritional data found'}
            
            # Store the enrichment data
            nutrition_id = self.store_nutrition_facts(product_id, nutrition_data)
            
            # Update product enrichment status
            self.update_product_enrichment(product_id, nutrition_data, nutrition_id)
            
            return {
                'status': 'success',
                'nutrition_id': nutrition_id,
                'data': nutrition_data,
                'search_query': search_query
            }
            
        except Exception as e:
            print(f"❌ Enrichment failed for {product.get('csv_product_name')}: {e}")
            return {'error': str(e)}
    
    def get_product_by_id(self, product_id: str) -> Dict:
        """Get product from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            product = conn.execute('''
                SELECT * FROM master_products WHERE id = ?
            ''', (product_id,)).fetchone()
            
            return dict(product) if product else {}
    
    def build_search_query(self, product: Dict) -> str:
        """Build optimized search query for Nutritionix"""
        # Use canonical name if available, otherwise product name
        base_name = product.get('canonical_name') or product.get('csv_product_name', '')
        brand = product.get('csv_brand', '')
        
        # Create search query - Nutritionix works better with simple terms
        if brand and brand.lower() not in ['generic', 'store brand', '']:
            query = f"{brand} {base_name}"
        else:
            query = base_name
        
        # Clean up the query
        query = query.replace('_', ' ').strip()
        
        # Handle specific cases for better matching
        query_mappings = {
            'banana': 'banana raw',
            'apple': 'apple raw with skin',
            'avocado': 'avocado raw',
            'strawberry': 'strawberries raw',
            'blueberry': 'blueberries raw',
            'raspberry': 'raspberries raw',
            'blackberry': 'blackberries raw'
        }
        
        for key, value in query_mappings.items():
            if key in query.lower():
                query = value
                break
        
        return query
    
    def search_nutrition(self, query: str) -> Optional[Dict]:
        """
        Search Nutritionix database for nutritional information
        """
        headers = {
            'x-app-id': self.app_id,
            'x-app-key': self.app_key,
            'Content-Type': 'application/json'
        }
        
        # First try natural language query
        natural_url = f"{self.base_url}/natural/nutrients"
        
        payload = {
            "query": query,
            "timezone": "US/Eastern"
        }
        
        try:
            response = requests.post(natural_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('foods') and len(data['foods']) > 0:
                    return self.parse_nutritionix_response(data['foods'][0])
            
            # If natural language fails, try search endpoint
            return self.search_by_item(query, headers)
            
        except requests.RequestException as e:
            print(f"Nutritionix API error: {e}")
            return None
    
    def search_by_item(self, query: str, headers: Dict) -> Optional[Dict]:
        """
        Use Nutritionix item search endpoint
        """
        search_url = f"{self.base_url}/search/instant"
        
        params = {
            'query': query
        }
        
        try:
            response = requests.get(search_url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Try common foods first
                if data.get('common') and len(data['common']) > 0:
                    food_name = data['common'][0]['food_name']
                    return self.get_food_details(food_name, headers)
                
                # Fall back to branded foods
                elif data.get('branded') and len(data['branded']) > 0:
                    nix_item_id = data['branded'][0]['nix_item_id']
                    return self.get_branded_food_details(nix_item_id, headers)
            
        except requests.RequestException as e:
            print(f"Nutritionix search error: {e}")
        
        return None
    
    def get_food_details(self, food_name: str, headers: Dict) -> Optional[Dict]:
        """Get detailed nutrition for common food"""
        natural_url = f"{self.base_url}/natural/nutrients"
        
        payload = {
            "query": food_name
        }
        
        try:
            response = requests.post(natural_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('foods') and len(data['foods']) > 0:
                    return self.parse_nutritionix_response(data['foods'][0])
                    
        except requests.RequestException as e:
            print(f"Food details error: {e}")
        
        return None
    
    def get_branded_food_details(self, nix_item_id: str, headers: Dict) -> Optional[Dict]:
        """Get detailed nutrition for branded food"""
        item_url = f"{self.base_url}/search/item"
        
        params = {
            'nix_item_id': nix_item_id
        }
        
        try:
            response = requests.get(item_url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('foods') and len(data['foods']) > 0:
                    return self.parse_nutritionix_response(data['foods'][0])
                    
        except requests.RequestException as e:
            print(f"Branded food details error: {e}")
        
        return None
    
    def parse_nutritionix_response(self, food_data: Dict) -> Dict:
        """
        Parse Nutritionix API response into standardized format
        """
        # Extract serving information
        serving_qty = food_data.get('serving_qty', 1)
        serving_unit = food_data.get('serving_unit', 'serving')
        serving_weight = food_data.get('serving_weight_grams', 0)
        
        # Extract macronutrients
        nutrition = {
            # Serving info
            'serving_size': serving_qty,
            'serving_unit': serving_unit,
            'serving_weight_grams': serving_weight,
            
            # Macronutrients
            'calories': food_data.get('nf_calories', 0),
            'total_fat': food_data.get('nf_total_fat', 0),
            'saturated_fat': food_data.get('nf_saturated_fat', 0),
            'trans_fat': food_data.get('nf_trans_fat', 0),
            'cholesterol': food_data.get('nf_cholesterol', 0),
            'sodium': food_data.get('nf_sodium', 0),
            'total_carbs': food_data.get('nf_total_carbohydrate', 0),
            'dietary_fiber': food_data.get('nf_dietary_fiber', 0),
            'sugars': food_data.get('nf_sugars', 0),
            'protein': food_data.get('nf_protein', 0),
            'potassium': food_data.get('nf_potassium', 0),
            
            # Additional nutritional info
            'vitamin_c': food_data.get('nf_vitamin_c', 0),
            'calcium': food_data.get('nf_calcium', 0),
            'iron': food_data.get('nf_iron', 0),
            'vitamin_a': food_data.get('nf_vitamin_a_iu', 0),
            
            # Product info
            'food_name': food_data.get('food_name', ''),
            'brand_name': food_data.get('brand_name'),
            'nutritionix_id': food_data.get('nix_item_id'),
            
            # Tags and categories
            'tags': food_data.get('tags', {}),
            'photo_url': food_data.get('photo', {}).get('thumb') if food_data.get('photo') else None
        }
        
        # Extract full nutrients if available
        if food_data.get('full_nutrients'):
            nutrient_mapping = {
                203: 'protein',
                204: 'total_fat', 
                205: 'total_carbs',
                208: 'calories',
                269: 'sugars',
                291: 'dietary_fiber',
                307: 'sodium',
                601: 'cholesterol',
                606: 'saturated_fat',
                605: 'trans_fat',
                306: 'potassium',
                301: 'calcium',
                303: 'iron',
                318: 'vitamin_a',
                401: 'vitamin_c'
            }
            
            for nutrient in food_data.get('full_nutrients', []):
                attr_id = nutrient.get('attr_id')
                value = nutrient.get('value', 0)
                
                if attr_id in nutrient_mapping:
                    nutrition[nutrient_mapping[attr_id]] = value
        
        return nutrition
    
    def store_nutrition_facts(self, product_id: str, nutrition_data: Dict) -> str:
        """Store detailed nutrition facts in database"""
        nutrition_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO nutrition_facts (
                    id, product_id, source, serving_size, serving_unit, serving_weight_grams,
                    calories, total_fat, saturated_fat, trans_fat, cholesterol, sodium,
                    total_carbs, dietary_fiber, sugars, protein, vitamin_a_iu, vitamin_c,
                    calcium, iron, potassium, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                nutrition_id, product_id, 'nutritionix',
                nutrition_data.get('serving_size', 1),
                nutrition_data.get('serving_unit', 'serving'),
                nutrition_data.get('serving_weight_grams', 0),
                nutrition_data.get('calories', 0),
                nutrition_data.get('total_fat', 0),
                nutrition_data.get('saturated_fat', 0),
                nutrition_data.get('trans_fat', 0),
                nutrition_data.get('cholesterol', 0),
                nutrition_data.get('sodium', 0),
                nutrition_data.get('total_carbs', 0),
                nutrition_data.get('dietary_fiber', 0),
                nutrition_data.get('sugars', 0),
                nutrition_data.get('protein', 0),
                nutrition_data.get('vitamin_a', 0),
                nutrition_data.get('vitamin_c', 0),
                nutrition_data.get('calcium', 0),
                nutrition_data.get('iron', 0),
                nutrition_data.get('potassium', 0),
                json.dumps(nutrition_data.get('tags', {}))
            ))
        
        return nutrition_id
    
    def update_product_enrichment(self, product_id: str, nutrition_data: Dict, nutrition_id: str):
        """Update product with enrichment information"""
        # Calculate new enrichment score
        enrichment_score = self.calculate_enrichment_score(nutrition_data)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE master_products 
                SET nutritionix_id = ?, nutrition_data = ?, enrichment_score = ?, 
                    last_enriched = ?, needs_enrichment = FALSE
                WHERE id = ?
            ''', (
                nutrition_data.get('nutritionix_id'),
                json.dumps(nutrition_data),
                enrichment_score,
                datetime.now().isoformat(),
                product_id
            ))
    
    def calculate_enrichment_score(self, nutrition_data: Dict) -> int:
        """Calculate enrichment score based on available nutritional data"""
        score = 25  # Base score for having some data
        
        # Macronutrients (40 points total)
        macros = ['calories', 'total_fat', 'total_carbs', 'protein']
        for macro in macros:
            if nutrition_data.get(macro, 0) > 0:
                score += 10
        
        # Micronutrients (20 points total) 
        micros = ['vitamin_c', 'calcium', 'iron', 'potassium']
        for micro in micros:
            if nutrition_data.get(micro, 0) > 0:
                score += 5
        
        # Detailed info (15 points total)
        details = ['dietary_fiber', 'sugars', 'sodium']
        for detail in details:
            if nutrition_data.get(detail, 0) >= 0:  # Sodium can be 0
                score += 5
        
        return min(score, 100)
    
    def enrich_all_products(self, batch_size: int = 5, delay: float = 1.0):
        """
        Enrich all products in database (batch processing)
        """
        # Get all products that need enrichment
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            products = conn.execute('''
                SELECT id, csv_product_name, canonical_name, csv_brand
                FROM master_products 
                WHERE needs_enrichment = TRUE OR last_enriched IS NULL
                ORDER BY created_at DESC
                LIMIT ?
            ''', (batch_size,)).fetchall()
        
        print(f"🍎 Starting enrichment for {len(products)} products...")
        
        results = {
            'success': 0,
            'errors': 0,
            'skipped': 0
        }
        
        for product in products:
            print(f"📊 Enriching: {product['csv_product_name']}...")
            
            result = self.enrich_product(product['id'])
            
            if result.get('status') == 'success':
                results['success'] += 1
                print(f"✅ Success: {product['csv_product_name']}")
            elif result.get('status') == 'already_enriched':
                results['skipped'] += 1  
                print(f"⚡ Skipped (already enriched): {product['csv_product_name']}")
            else:
                results['errors'] += 1
                print(f"❌ Error: {product['csv_product_name']} - {result.get('error')}")
            
            # Rate limiting
            if delay > 0:
                time.sleep(delay)
        
        print(f"\n🎉 Enrichment Complete:")
        print(f"   ✅ Success: {results['success']}")
        print(f"   ⚡ Skipped: {results['skipped']}")
        print(f"   ❌ Errors: {results['errors']}")
        
        return results
    
    def get_product_nutrition(self, product_id: str) -> Optional[Dict]:
        """Get nutrition facts for a product"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            nutrition = conn.execute('''
                SELECT * FROM nutrition_facts WHERE product_id = ? ORDER BY created_at DESC LIMIT 1
            ''', (product_id,)).fetchone()
            
            return dict(nutrition) if nutrition else None

def main():
    """CLI interface for testing Nutritionix integration"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python nutritionix_enricher.py <database_path> [app_id] [app_key]")
        print("\nCommands:")
        print("  enrich_all - Enrich all products")
        print("  enrich <product_id> - Enrich specific product")
        return
    
    db_path = sys.argv[1]
    
    # For demo purposes, use mock credentials
    enricher = NutritionixEnricher(
        db_path,
        app_id=sys.argv[2] if len(sys.argv) > 2 else "demo_app_id",
        app_key=sys.argv[3] if len(sys.argv) > 3 else "demo_app_key"
    )
    
    if len(sys.argv) > 4 and sys.argv[4] == "enrich_all":
        enricher.enrich_all_products(batch_size=5)
    else:
        print("🍎 Nutritionix Enricher Setup Complete")
        print("📊 Database schema updated for nutritional data")
        print("⚠️  Note: Add your real Nutritionix API credentials to enable enrichment")

if __name__ == "__main__":
    main()