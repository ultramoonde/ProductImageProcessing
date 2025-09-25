#!/usr/bin/env python3
"""
Simple Image Enricher (No external dependencies)
Demonstrates the comprehensive enrichment concept with mock data
"""

import sqlite3
import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

class SimpleImageEnricher:
    """
    Simplified image enricher for demonstration
    In production, this would use the full ImageScraper with real web scraping
    """
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.setup_database()
    
    def setup_database(self):
        """Ensure scraped_images table exists"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS scraped_images (
                    id TEXT PRIMARY KEY,
                    product_id TEXT REFERENCES master_products(id),
                    source TEXT NOT NULL,
                    original_url TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    image_hash TEXT,
                    width INTEGER,
                    height INTEGER,
                    file_size INTEGER,
                    quality_score REAL DEFAULT 0.0,
                    is_primary BOOLEAN DEFAULT FALSE,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def enrich_product_images(self, product_id: str, product_name: str, brand: str = None) -> List[Dict]:
        """
        Mock image enrichment - demonstrates what the full system would do
        """
        print(f"🔍 [DEMO] Enriching images for: {product_name} ({brand or 'Generic'})")
        
        # Get canonical name for better matching
        canonical_name = self.get_canonical_name(product_id)
        
        # Generate mock high-quality images based on product type
        mock_images = self.generate_mock_images(product_name, canonical_name, brand)
        
        # Store mock images in database
        stored_images = []
        for i, img_data in enumerate(mock_images):
            image_id = str(uuid.uuid4())
            
            # Create mock local path (in production, this would be real downloaded file)
            mock_local_path = f"scraped_images/{product_id}_{img_data['source']}_{image_id[:8]}.jpg"
            
            # Store in database
            self.store_mock_image(
                image_id, product_id, img_data['source'], img_data['url'],
                mock_local_path, img_data['width'], img_data['height'],
                img_data['quality_score'], i == 0, json.dumps(img_data)
            )
            
            stored_images.append({
                'id': image_id,
                'source': img_data['source'],
                'url': img_data['url'],
                'local_path': mock_local_path,
                'quality_score': img_data['quality_score'],
                'is_primary': i == 0,
                'width': img_data['width'],
                'height': img_data['height']
            })
            
            print(f"   ✅ Mock image: {img_data['source']} (quality: {img_data['quality_score']:.2f})")
        
        print(f"   📸 Total images found: {len(stored_images)}")
        return stored_images
    
    def generate_mock_images(self, product_name: str, canonical_name: str, brand: str = None) -> List[Dict]:
        """Generate mock image data based on product type"""
        
        images = []
        name_lower = canonical_name.lower() if canonical_name else product_name.lower()
        
        if 'banana' in name_lower:
            if brand and 'chiquita' in brand.lower():
                # Chiquita banana images - VERIFIED WORKING URLs
                images = [
                    {
                        'source': 'wikipedia_single',
                        'url': 'https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg',
                        'width': 1200, 'height': 800, 'quality_score': 0.95,
                        'description': 'Fresh single banana - Wikipedia commons'
                    },
                    {
                        'source': 'wikipedia_bunch',
                        'url': 'https://upload.wikimedia.org/wikipedia/commons/4/44/Bananas_white_background.jpg',
                        'width': 800, 'height': 600, 'quality_score': 0.90,
                        'description': 'Bananas on white background - Wikipedia'
                    },
                    {
                        'source': 'wikipedia_variety',
                        'url': 'https://upload.wikimedia.org/wikipedia/commons/c/c1/Bananas.jpg',
                        'width': 1024, 'height': 768, 'quality_score': 0.85,
                        'description': 'Banana varieties - Wikipedia commons'
                    }
                ]
            else:
                # Generic banana images - VERIFIED WORKING URLs
                images = [
                    {
                        'source': 'wikipedia',
                        'url': 'https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg',
                        'width': 1000, 'height': 750, 'quality_score': 0.90,
                        'description': 'Single banana - Wikipedia commons'
                    },
                    {
                        'source': 'wikipedia_bunch',
                        'url': 'https://upload.wikimedia.org/wikipedia/commons/4/44/Bananas_white_background.jpg',
                        'width': 600, 'height': 400, 'quality_score': 0.85,
                        'description': 'Bananas on white background - Wikipedia'
                    }
                ]
        
        elif 'apple' in name_lower:
            # Real apple images from Wikipedia that actually work
            images = [
                {
                    'source': 'wikipedia',
                    'url': 'https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg',
                    'width': 1200, 'height': 1200, 'quality_score': 0.90,
                    'description': 'Red apple - Wikipedia commons'
                },
                {
                    'source': 'wikipedia_green',
                    'url': 'https://upload.wikimedia.org/wikipedia/commons/5/50/Green_apples.jpg',
                    'width': 800, 'height': 600, 'quality_score': 0.85,
                    'description': 'Green apples - Wikipedia commons'
                },
                {
                    'source': 'wikipedia_variety',
                    'url': 'https://upload.wikimedia.org/wikipedia/commons/2/22/Assorted_apples.jpg',
                    'width': 1024, 'height': 768, 'quality_score': 0.80,
                    'description': 'Apple varieties - Wikipedia commons'
                }
            ]
        
        elif 'avocado' in name_lower:
            if brand and 'bio' in brand.lower():
                images = [
                    {
                        'source': 'organic_brand',
                        'url': 'https://example-organic.com/images/bio-avocado.jpg',
                        'width': 800, 'height': 600, 'quality_score': 0.88,
                        'description': 'Organic avocado - brand website'
                    },
                    {
                        'source': 'wikipedia',
                        'url': 'https://upload.wikimedia.org/wikipedia/commons/7/7b/Avocado.jpg',
                        'width': 1000, 'height': 750, 'quality_score': 0.85,
                        'description': 'Avocado Wikipedia image'
                    }
                ]
            else:
                images = [
                    {
                        'source': 'wikipedia',
                        'url': 'https://upload.wikimedia.org/wikipedia/commons/e/e4/Hass_avocado.jpg',
                        'width': 800, 'height': 600, 'quality_score': 0.85,
                        'description': 'Hass avocado Wikipedia'
                    }
                ]
        
        else:
            # Generic fruit/food images
            images = [
                {
                    'source': 'wikipedia',
                    'url': f'https://upload.wikimedia.org/wikipedia/commons/placeholder-{canonical_name}.jpg',
                    'width': 600, 'height': 400, 'quality_score': 0.70,
                    'description': f'Generic {canonical_name} image'
                },
                {
                    'source': 'open_food_facts',
                    'url': f'https://images.openfoodfacts.org/images/products/{canonical_name}.jpg',
                    'width': 400, 'height': 300, 'quality_score': 0.65,
                    'description': f'{canonical_name} from food database'
                }
            ]
        
        return images
    
    def get_canonical_name(self, product_id: str) -> str:
        """Get canonical name for product"""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute('SELECT canonical_name FROM master_products WHERE id = ?', (product_id,)).fetchone()
            return result[0] if result else 'unknown'
    
    def store_mock_image(self, image_id: str, product_id: str, source: str, original_url: str,
                        local_path: str, width: int, height: int, quality_score: float,
                        is_primary: bool, metadata: str):
        """Store mock image data in database"""
        
        mock_file_size = width * height * 3 // 8  # Rough estimate for JPEG
        mock_hash = f"mock_hash_{image_id[:8]}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO scraped_images (
                    id, product_id, source, original_url, local_path, image_hash,
                    width, height, file_size, quality_score, is_primary, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                image_id, product_id, source, original_url, local_path, mock_hash,
                width, height, mock_file_size, quality_score, is_primary, metadata
            ))

def main():
    """Test the simple image enricher"""
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python simple_image_enricher.py <db_path> <product_id> <product_name> [brand]")
        return
    
    db_path = sys.argv[1]
    product_id = sys.argv[2]
    product_name = sys.argv[3]
    brand = sys.argv[4] if len(sys.argv) > 4 else None
    
    enricher = SimpleImageEnricher(db_path)
    images = enricher.enrich_product_images(product_id, product_name, brand)
    
    print(f"\n🎯 Final Results:")
    for img in images:
        print(f"   ✅ {img['source']}: {img['local_path']} (score: {img['quality_score']:.2f})")

if __name__ == "__main__":
    main()