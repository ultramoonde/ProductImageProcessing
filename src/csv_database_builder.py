#!/usr/bin/env python3
"""
CSV Product Database Builder
Transforms user CSV + images into master product database with full enrichment
"""

import pandas as pd
import sqlite3
import asyncio
import json
import hashlib
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import aiohttp
import requests
import ssl
from difflib import SequenceMatcher

class CSVProductDatabaseBuilder:
    """
    Transforms CSV + images into enriched master database
    """
    
    def __init__(self, csv_path: str, image_base_path: str):
        self.csv_path = Path(csv_path)
        self.image_base_path = Path(image_base_path)
        self.db_path = Path("master_product_database.db")
        
        # Data sources for enrichment
        self.data_sources = {
            'openfoodfacts': 'https://world.openfoodfacts.org/api/v0/product/',
            'usda': 'https://api.nal.usda.gov/fdc/v1/foods/search',
            'edamam': 'https://api.edamam.com/api/food-database/v2/parser'
        }
        
        self.setup_database()
    
    def setup_database(self):
        """Create master product database schema"""
        with sqlite3.connect(self.db_path) as conn:
            # Master products table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS master_products (
                    id TEXT PRIMARY KEY,
                    product_signature TEXT UNIQUE,
                    
                    -- CSV source data
                    csv_uid TEXT,
                    csv_price TEXT,
                    csv_brand TEXT,
                    csv_product_name TEXT,
                    csv_weight TEXT,
                    csv_extracted_text TEXT,
                    csv_image_path TEXT,
                    
                    -- Canonical enriched data
                    canonical_name TEXT,
                    canonical_brand TEXT,
                    canonical_category TEXT,
                    
                    -- Nutrition data (JSON)
                    nutrition_per_100g TEXT, -- JSON
                    ingredients_list TEXT,
                    allergens TEXT, -- JSON array
                    dietary_flags TEXT, -- JSON array
                    
                    -- Quality metrics
                    enrichment_score INTEGER DEFAULT 0,
                    source_count INTEGER DEFAULT 0,
                    confidence_score REAL DEFAULT 0.0,
                    
                    -- Timestamps
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_enriched TIMESTAMP,
                    needs_refresh BOOLEAN DEFAULT TRUE
                )
            ''')
            
            # Source tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS enrichment_sources (
                    id TEXT PRIMARY KEY,
                    product_id TEXT REFERENCES master_products(id),
                    source_name TEXT,
                    source_url TEXT,
                    data_type TEXT, -- 'nutrition', 'images', 'ingredients'
                    raw_data TEXT, -- JSON
                    confidence REAL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Image assets
            conn.execute('''
                CREATE TABLE IF NOT EXISTS product_images (
                    id TEXT PRIMARY KEY,
                    product_id TEXT REFERENCES master_products(id),
                    image_type TEXT, -- 'csv_original', 'high_quality', 'nutrition_label'
                    image_path TEXT,
                    image_url TEXT,
                    quality_score REAL DEFAULT 0.0,
                    width INTEGER,
                    height INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def create_product_signature(self, name: str, brand: str, weight: str) -> str:
        """Create unique product signature for deduplication"""
        # Normalize inputs
        norm_name = self.normalize_product_name(name)
        norm_brand = str(brand).lower().strip() if brand and not pd.isna(brand) else ""
        norm_weight = self.normalize_weight(weight)
        
        # Create signature
        signature_text = f"{norm_brand}_{norm_name}_{norm_weight}"
        return hashlib.md5(signature_text.encode()).hexdigest()[:16]
    
    def normalize_product_name(self, name) -> str:
        """Normalize product names for better matching"""
        if not name or pd.isna(name):
            return ""
        name = str(name)  # Ensure it's a string
        
        # Common normalizations for German grocery items
        normalizations = {
            'bananen': 'banana',
            'apfel': 'apple', 
            'heidelbeeren': 'blueberries',
            'himbeeren': 'raspberries',
            'brombeeren': 'blackberries',
            'erdbeeren': 'strawberries',
            'avocado': 'avocado',
            'ananas': 'pineapple',
            'birne': 'pear'
        }
        
        name_lower = name.lower().strip()
        for german, english in normalizations.items():
            if german in name_lower:
                return english
        
        return name_lower.replace(' ', '_')
    
    def normalize_weight(self, weight) -> str:
        """Normalize weight/quantity strings"""
        if not weight or pd.isna(weight):
            return ""
        weight = str(weight)  # Ensure it's a string
        
        weight_lower = weight.lower().replace(' ', '')
        
        # Convert to standard units
        if 'stk' in weight_lower or 'stück' in weight_lower:
            # Extract number of pieces
            import re
            match = re.search(r'(\d+)', weight_lower)
            if match:
                return f"{match.group(1)}pcs"
        elif 'kg' in weight_lower:
            return weight_lower.replace('ikg', '1kg')
        elif 'g' in weight_lower:
            return weight_lower
            
        return weight_lower
    
    async def process_csv_products(self):
        """Main processing pipeline for CSV products"""
        print("🚀 Starting CSV product database builder...")
        
        # Load CSV data
        df = pd.read_csv(self.csv_path)
        print(f"📊 Loaded {len(df)} products from CSV")
        
        processed_count = 0
        enriched_count = 0
        
        for idx, row in df.iterrows():
            try:
                # Process each product
                product_id = await self.process_single_product(row)
                if product_id:
                    processed_count += 1
                    
                    # Try to enrich from external sources
                    if await self.enrich_product(product_id, row):
                        enriched_count += 1
                        
                print(f"✅ Processed {processed_count}/{len(df)} - Enriched: {enriched_count}")
                
            except Exception as e:
                print(f"❌ Error processing {row.get('product_name', 'unknown')}: {e}")
        
        print(f"🎉 Complete! Processed {processed_count} products, enriched {enriched_count}")
        return processed_count, enriched_count
    
    async def process_single_product(self, row: pd.Series) -> str:
        """Process a single CSV row into database"""
        
        # Create product signature for deduplication
        signature = self.create_product_signature(
            row.get('product_name', ''),
            row.get('brand', ''),
            row.get('weight', '')
        )
        
        # Check if product already exists
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                'SELECT id FROM master_products WHERE product_signature = ?',
                (signature,)
            ).fetchone()
            
            if existing:
                print(f"⚡ Found existing: {row.get('product_name')} - skipping")
                return existing[0]
        
        # Create new product record
        product_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO master_products (
                    id, product_signature, csv_uid, csv_price, csv_brand,
                    csv_product_name, csv_weight, csv_extracted_text, csv_image_path,
                    canonical_name, canonical_brand
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                product_id,
                signature,
                row.get('uid', ''),
                row.get('price', ''),
                row.get('brand', ''),
                row.get('product_name', ''),
                row.get('weight', ''),
                row.get('all_extracted_text', ''),
                row.get('tile_image_path', ''),
                self.normalize_product_name(row.get('product_name', '')),
                row.get('brand', '').strip() if row.get('brand') else None
            ))
        
        # Store original image reference
        if row.get('tile_image_path'):
            await self.store_image_reference(product_id, row.get('tile_image_path'), 'csv_original')
        
        print(f"📦 Created product: {row.get('product_name')} (ID: {product_id[:8]})")
        return product_id
    
    async def store_image_reference(self, product_id: str, image_path: str, image_type: str):
        """Store image reference in database"""
        
        full_path = self.image_base_path / image_path
        
        # Get image dimensions if file exists
        width, height = 0, 0
        if full_path.exists():
            try:
                from PIL import Image
                with Image.open(full_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"⚠️  Could not read image dimensions: {e}")
        
        image_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO product_images (
                    id, product_id, image_type, image_path, width, height, quality_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                image_id,
                product_id, 
                image_type,
                str(full_path),
                width,
                height,
                0.7 if full_path.exists() else 0.0
            ))
    
    async def enrich_product(self, product_id: str, row: pd.Series) -> bool:
        """Enrich product with data from external sources"""
        
        canonical_name = self.normalize_product_name(row.get('product_name', ''))
        brand = row.get('brand', '')
        
        enrichment_success = False
        
        # Try Open Food Facts first (works well for European products)
        if await self.enrich_from_openfoodfacts(product_id, canonical_name, brand):
            enrichment_success = True
        
        # Try USDA database (good for basic fruits/vegetables)
        if await self.enrich_from_usda(product_id, canonical_name):
            enrichment_success = True
        
        # Update enrichment score
        if enrichment_success:
            self.update_enrichment_score(product_id)
        
        return enrichment_success
    
    async def enrich_from_openfoodfacts(self, product_id: str, canonical_name: str, brand: str) -> bool:
        """Enrich from Open Food Facts database"""
        try:
            # Search for product
            search_url = f"https://world.openfoodfacts.org/cgi/search.pl"
            params = {
                'search_terms': f"{canonical_name} {brand}".strip(),
                'search_simple': 1,
                'json': 1,
                'page_size': 3
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('products'):
                            # Take the first result
                            product = data['products'][0]
                            
                            # Extract nutrition data
                            nutrition = {}
                            nutrients = product.get('nutriments', {})
                            
                            for key, value in nutrients.items():
                                if key.endswith('_100g') and isinstance(value, (int, float)):
                                    nutrition[key] = value
                            
                            # Store enrichment data
                            await self.store_enrichment_data(
                                product_id,
                                'openfoodfacts',
                                f"https://world.openfoodfacts.org/product/{product.get('code', '')}",
                                'nutrition',
                                {
                                    'nutrition_per_100g': nutrition,
                                    'ingredients': product.get('ingredients_text', ''),
                                    'allergens': product.get('allergens', '').split(',') if product.get('allergens') else [],
                                    'image_url': product.get('image_url', ''),
                                    'brand': product.get('brands', ''),
                                    'categories': product.get('categories', '')
                                },
                                0.8
                            )
                            
                            print(f"🥗 Enriched from OpenFoodFacts: {canonical_name}")
                            return True
                            
        except Exception as e:
            print(f"❌ OpenFoodFacts enrichment failed: {e}")
        
        return False
    
    async def enrich_from_usda(self, product_id: str, canonical_name: str) -> bool:
        """Enrich from USDA Food Database (free, no API key needed)"""
        try:
            # USDA search endpoint
            search_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
            params = {
                'query': canonical_name,
                'pageSize': 3,
                'dataType': ['Foundation', 'SR Legacy']
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('foods'):
                            # Take first result
                            food = data['foods'][0]
                            
                            # Extract nutrition data
                            nutrition = {}
                            for nutrient in food.get('foodNutrients', []):
                                name = nutrient.get('nutrientName', '').lower()
                                value = nutrient.get('value')
                                
                                if value and isinstance(value, (int, float)):
                                    if 'energy' in name or 'caloric' in name:
                                        nutrition['energy_kcal_100g'] = value
                                    elif 'protein' in name:
                                        nutrition['proteins_100g'] = value
                                    elif 'carbohydrate' in name:
                                        nutrition['carbohydrates_100g'] = value
                                    elif 'total lipid' in name or 'fat' in name:
                                        nutrition['fat_100g'] = value
                                    elif 'fiber' in name:
                                        nutrition['fiber_100g'] = value
                                    elif 'sugars' in name:
                                        nutrition['sugars_100g'] = value
                            
                            # Store enrichment data  
                            await self.store_enrichment_data(
                                product_id,
                                'usda',
                                f"https://fdc.nal.usda.gov/fdc-app.html#/food-details/{food.get('fdcId')}",
                                'nutrition',
                                {
                                    'nutrition_per_100g': nutrition,
                                    'description': food.get('description', ''),
                                    'food_category': food.get('foodCategory', ''),
                                    'scientific_name': food.get('scientificName', '')
                                },
                                0.9
                            )
                            
                            print(f"🇺🇸 Enriched from USDA: {canonical_name}")
                            return True
                            
        except Exception as e:
            print(f"❌ USDA enrichment failed: {e}")
        
        return False
    
    async def store_enrichment_data(self, product_id: str, source_name: str, 
                                  source_url: str, data_type: str, data: Dict, confidence: float):
        """Store enrichment data from external source"""
        
        source_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO enrichment_sources (
                    id, product_id, source_name, source_url, data_type, raw_data, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                source_id,
                product_id,
                source_name,
                source_url,
                data_type,
                json.dumps(data),
                confidence
            ))
            
            # Update master product with nutrition data
            if data_type == 'nutrition' and 'nutrition_per_100g' in data:
                conn.execute('''
                    UPDATE master_products 
                    SET nutrition_per_100g = ?, 
                        ingredients_list = ?,
                        allergens = ?,
                        last_enriched = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (
                    json.dumps(data['nutrition_per_100g']),
                    data.get('ingredients', ''),
                    json.dumps(data.get('allergens', [])),
                    product_id
                ))
    
    def update_enrichment_score(self, product_id: str):
        """Calculate and update enrichment score"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Count sources and calculate score
            sources = conn.execute('''
                SELECT COUNT(*), AVG(confidence) FROM enrichment_sources 
                WHERE product_id = ?
            ''', (product_id,)).fetchone()
            
            source_count, avg_confidence = sources[0], sources[1] or 0.0
            
            # Simple scoring: source count * average confidence * 100
            enrichment_score = min(100, int(source_count * avg_confidence * 50))
            
            conn.execute('''
                UPDATE master_products 
                SET source_count = ?, confidence_score = ?, enrichment_score = ?
                WHERE id = ?
            ''', (source_count, avg_confidence, enrichment_score, product_id))
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Product counts
            total_products = conn.execute('SELECT COUNT(*) FROM master_products').fetchone()[0]
            enriched_products = conn.execute('SELECT COUNT(*) FROM master_products WHERE enrichment_score > 0').fetchone()[0]
            
            # Average enrichment score
            avg_score = conn.execute('SELECT AVG(enrichment_score) FROM master_products').fetchone()[0] or 0
            
            # Source breakdown
            sources = conn.execute('''
                SELECT source_name, COUNT(*) FROM enrichment_sources 
                GROUP BY source_name
            ''').fetchall()
            
            return {
                'total_products': total_products,
                'enriched_products': enriched_products,
                'enrichment_rate': f"{(enriched_products/total_products*100):.1f}%" if total_products > 0 else "0%",
                'average_score': f"{avg_score:.1f}",
                'sources': dict(sources)
            }

# CLI Interface
async def main():
    """Main CLI interface"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python csv_database_builder.py <csv_path> [image_base_path]")
        return
    
    csv_path = sys.argv[1]
    image_base_path = sys.argv[2] if len(sys.argv) > 2 else str(Path(csv_path).parent)
    
    builder = CSVProductDatabaseBuilder(csv_path, image_base_path)
    
    print("🚀 Building master product database from CSV...")
    processed, enriched = await builder.process_csv_products()
    
    stats = builder.get_database_stats()
    print("\n📊 Database Statistics:")
    print(f"   Total Products: {stats['total_products']}")
    print(f"   Enriched: {stats['enriched_products']} ({stats['enrichment_rate']})")
    print(f"   Average Score: {stats['average_score']}/100")
    print(f"   Data Sources: {stats['sources']}")
    print(f"\n💾 Database created: {builder.db_path}")

if __name__ == "__main__":
    asyncio.run(main())