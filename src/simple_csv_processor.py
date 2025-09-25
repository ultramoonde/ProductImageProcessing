#!/usr/bin/env python3
"""
Simple CSV Product Processor
Process user CSV + images into database without external API dependencies
"""

import pandas as pd
import sqlite3
import json
import hashlib
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class SimpleCSVProcessor:
    """
    Simple processor that creates database from CSV + local analysis
    """
    
    def __init__(self, csv_path: str, image_base_path: str):
        self.csv_path = Path(csv_path)
        self.image_base_path = Path(image_base_path)
        self.db_path = Path("master_product_database.db")
        self.setup_database()
    
    def setup_database(self):
        """Create master product database schema"""
        with sqlite3.connect(self.db_path) as conn:
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
                    csv_quantity TEXT,
                    csv_price_per_unit TEXT,
                    csv_extracted_text TEXT,
                    csv_image_path TEXT,
                    
                    -- Normalized data
                    canonical_name TEXT,
                    canonical_brand TEXT,
                    canonical_category TEXT,
                    normalized_price_eur REAL,
                    normalized_weight TEXT,
                    
                    -- Enrichment status
                    enrichment_score INTEGER DEFAULT 0,
                    needs_enrichment BOOLEAN DEFAULT TRUE,
                    
                    -- Timestamps
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS product_images (
                    id TEXT PRIMARY KEY,
                    product_id TEXT REFERENCES master_products(id),
                    image_type TEXT,
                    image_path TEXT,
                    file_exists BOOLEAN,
                    width INTEGER,
                    height INTEGER
                )
            ''')
    
    def normalize_german_product(self, name, brand):
        """Normalize German grocery product names"""
        if pd.isna(name):
            name = ""
        else:
            name = str(name)
            
        if pd.isna(brand):
            brand = ""
        else:
            brand = str(brand)
        
        # German to English mappings for better database consistency
        translations = {
            'bananen': 'banana',
            'apfel': 'apple',
            'heidelbeeren': 'blueberries', 
            'himbeeren': 'raspberries',
            'brombeeren': 'blackberries',
            'erdbeeren': 'strawberries',
            'avocado': 'avocado',
            'ananas': 'pineapple',
            'birne': 'pear',
            'obst': 'fruit',
            'süsse': 'sweet'
        }
        
        canonical_name = name.lower()
        for german, english in translations.items():
            if german in canonical_name:
                canonical_name = english
                break
        
        # Category detection
        fruit_indicators = ['banana', 'apple', 'blueberries', 'strawberries', 'avocado', 'pineapple', 'pear']
        category = 'fruit' if any(fruit in canonical_name for fruit in fruit_indicators) else 'grocery'
        
        return canonical_name, category
    
    def parse_price(self, price_str):
        """Parse German price format to float"""
        if pd.isna(price_str):
            return 0.0
            
        price_str = str(price_str).replace('€', '').replace(',', '.').strip()
        
        try:
            return float(price_str)
        except:
            return 0.0
    
    def create_product_signature(self, name, brand, weight):
        """Create unique signature for deduplication"""
        name_norm = str(name).lower().strip() if name and not pd.isna(name) else ""
        brand_norm = str(brand).lower().strip() if brand and not pd.isna(brand) else ""  
        weight_norm = str(weight).lower().strip() if weight and not pd.isna(weight) else ""
        
        signature_text = f"{brand_norm}_{name_norm}_{weight_norm}"
        return hashlib.md5(signature_text.encode()).hexdigest()[:16]
    
    def process_csv(self):
        """Process the CSV file and create database"""
        print("🚀 Processing CSV into master product database...")
        
        # Load CSV with proper handling of missing values
        df = pd.read_csv(self.csv_path, keep_default_na=False, na_values=[''])
        print(f"📊 Loaded {len(df)} products from CSV")
        
        processed_count = 0
        skipped_count = 0
        
        for idx, row in df.iterrows():
            try:
                # Create product signature for deduplication
                signature = self.create_product_signature(
                    row.get('product_name'),
                    row.get('brand'), 
                    row.get('weight')
                )
                
                # Check if already exists
                with sqlite3.connect(self.db_path) as conn:
                    existing = conn.execute(
                        'SELECT id FROM master_products WHERE product_signature = ?', 
                        (signature,)
                    ).fetchone()
                    
                    if existing:
                        print(f"⚡ Duplicate found: {row.get('product_name')} - skipping")
                        skipped_count += 1
                        continue
                
                # Process new product
                product_id = str(uuid.uuid4())
                
                # Normalize product data
                canonical_name, category = self.normalize_german_product(
                    row.get('product_name'), row.get('brand')
                )
                
                price_eur = self.parse_price(row.get('price'))
                
                # Insert product
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT INTO master_products (
                            id, product_signature, csv_uid, csv_price, csv_brand,
                            csv_product_name, csv_weight, csv_quantity, csv_price_per_unit,
                            csv_extracted_text, csv_image_path,
                            canonical_name, canonical_brand, canonical_category,
                            normalized_price_eur, normalized_weight, enrichment_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        product_id, signature,
                        row.get('uid', ''), row.get('price', ''), row.get('brand', ''),
                        row.get('product_name', ''), row.get('weight', ''), row.get('quantity', ''),
                        row.get('price_per_unit', ''), row.get('all_extracted_text', ''),
                        row.get('tile_image_path', ''),
                        canonical_name, row.get('brand', ''), category,
                        price_eur, row.get('weight', ''), 25  # Basic enrichment score
                    ))
                
                # Process image if exists
                if row.get('tile_image_path'):
                    self.process_image(product_id, row.get('tile_image_path'))
                
                processed_count += 1
                print(f"📦 Processed: {row.get('product_name')} ({canonical_name})")
                
            except Exception as e:
                print(f"❌ Error processing row {idx}: {e}")
        
        print(f"\n🎉 Processing complete!")
        print(f"   ✅ Processed: {processed_count}")  
        print(f"   ⚡ Skipped duplicates: {skipped_count}")
        print(f"   💾 Database: {self.db_path}")
        
        return self.get_database_stats()
    
    def process_image(self, product_id: str, image_path: str):
        """Process and store image reference"""
        full_path = self.image_base_path / image_path
        
        exists = full_path.exists()
        width, height = 0, 0
        
        if exists:
            try:
                from PIL import Image
                with Image.open(full_path) as img:
                    width, height = img.size
            except Exception:
                pass
        
        image_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO product_images (
                    id, product_id, image_type, image_path, file_exists, width, height
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                image_id, product_id, 'product_tile', 
                str(full_path), exists, width, height
            ))
    
    def get_database_stats(self):
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute('SELECT COUNT(*) FROM master_products').fetchone()[0]
            
            categories = conn.execute('''
                SELECT canonical_category, COUNT(*) 
                FROM master_products 
                GROUP BY canonical_category
            ''').fetchall()
            
            brands = conn.execute('''
                SELECT csv_brand, COUNT(*) 
                FROM master_products 
                WHERE csv_brand != ""
                GROUP BY csv_brand
                ORDER BY COUNT(*) DESC
            ''').fetchall()
            
            avg_price = conn.execute('''
                SELECT AVG(normalized_price_eur) 
                FROM master_products 
                WHERE normalized_price_eur > 0
            ''').fetchone()[0]
            
            images = conn.execute('''
                SELECT COUNT(*) FROM product_images WHERE file_exists = 1
            ''').fetchone()[0]
            
            return {
                'total_products': total,
                'categories': dict(categories),
                'top_brands': brands[:5],
                'average_price': f"€{avg_price:.2f}" if avg_price else "€0.00",
                'images_found': images
            }

def main():
    """CLI interface"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python simple_csv_processor.py <csv_path> [image_base_path]")
        return
    
    csv_path = sys.argv[1]
    image_base_path = sys.argv[2] if len(sys.argv) > 2 else str(Path(csv_path).parent)
    
    processor = SimpleCSVProcessor(csv_path, image_base_path)
    stats = processor.process_csv()
    
    print(f"\n📊 Database Statistics:")
    print(f"   Total Products: {stats['total_products']}")
    print(f"   Categories: {stats['categories']}")
    print(f"   Top Brands: {[f'{brand} ({count})' for brand, count in stats['top_brands']]}")
    print(f"   Average Price: {stats['average_price']}")
    print(f"   Images Found: {stats['images_found']}")

if __name__ == "__main__":
    main()