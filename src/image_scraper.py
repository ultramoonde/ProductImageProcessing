#!/usr/bin/env python3
"""
Comprehensive Image Scraper for Food Products
Automatically finds and downloads high-quality product images from multiple sources
"""

import requests
from bs4 import BeautifulSoup
import sqlite3
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
import json
import hashlib
from urllib.parse import urljoin, urlparse
import re
from datetime import datetime

class ImageScraper:
    """
    Multi-source image scraper for food products
    Sources: Wikipedia, brand websites, Open Images, product databases
    """
    
    def __init__(self, db_path: str, images_dir: str = "scraped_images"):
        self.db_path = Path(db_path)
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(exist_ok=True)
        self.setup_database()
        
        # User agents for different scraping scenarios
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    def setup_database(self):
        """Extend database schema for scraped images"""
        with sqlite3.connect(self.db_path) as conn:
            # Add scraped_images table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS scraped_images (
                    id TEXT PRIMARY KEY,
                    product_id TEXT REFERENCES master_products(id),
                    source TEXT NOT NULL,  -- 'wikipedia', 'brand_website', 'open_images', etc.
                    original_url TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    image_hash TEXT,
                    width INTEGER,
                    height INTEGER,
                    file_size INTEGER,
                    quality_score REAL DEFAULT 0.0,
                    is_primary BOOLEAN DEFAULT FALSE,
                    metadata TEXT,  -- JSON with additional info
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def scrape_product_images(self, product_id: str, product_name: str, brand: str = None) -> List[Dict]:
        """
        Main orchestrator - scrapes images from all available sources
        """
        print(f"🔍 Scraping images for: {product_name} ({brand or 'no brand'})")
        
        all_images = []
        
        # 1. Wikipedia images (highest quality, most reliable)
        wikipedia_images = self.scrape_wikipedia_images(product_name, brand)
        all_images.extend(wikipedia_images)
        
        # 2. Brand website images (if brand exists)
        if brand:
            brand_images = self.scrape_brand_website(product_name, brand)
            all_images.extend(brand_images)
        
        # 3. Open Food Facts database
        off_images = self.scrape_open_food_facts(product_name, brand)
        all_images.extend(off_images)
        
        # 4. Google Images (as fallback - be careful with rate limits)
        if len(all_images) < 2:  # Only if we don't have enough images
            google_images = self.scrape_google_images(product_name, brand)
            all_images.extend(google_images)
        
        # Score and rank images
        scored_images = self.score_and_rank_images(all_images, product_name, brand)
        
        # Download and store best images
        final_images = self.download_and_store_images(product_id, scored_images[:5])  # Top 5
        
        return final_images
    
    def scrape_wikipedia_images(self, product_name: str, brand: str = None) -> List[Dict]:
        """Scrape high-quality images from Wikipedia"""
        images = []
        
        try:
            # Search Wikipedia for the product
            search_terms = [
                product_name,
                f"{product_name} fruit" if not brand else f"{brand} {product_name}",
                f"{product_name} food"
            ]
            
            for term in search_terms:
                try:
                    # Wikipedia API search
                    search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{term.replace(' ', '_')}"
                    response = requests.get(search_url, headers=self.headers, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Get main image
                        if 'originalimage' in data:
                            images.append({
                                'url': data['originalimage']['source'],
                                'source': 'wikipedia_main',
                                'width': data['originalimage']['width'],
                                'height': data['originalimage']['height'],
                                'title': data.get('title', term),
                                'description': data.get('extract', ''),
                                'quality_hint': 'high'  # Wikipedia images are usually high quality
                            })
                        
                        # Get additional images from the page
                        if 'pageid' in data:
                            page_images = self.get_wikipedia_page_images(data['pageid'])
                            images.extend(page_images)
                        
                        break  # Found a good Wikipedia page
                        
                except Exception as e:
                    print(f"   ⚠️ Wikipedia search failed for '{term}': {e}")
                    continue
                    
        except Exception as e:
            print(f"   ❌ Wikipedia scraping error: {e}")
        
        print(f"   📸 Found {len(images)} Wikipedia images")
        return images
    
    def get_wikipedia_page_images(self, page_id: int) -> List[Dict]:
        """Get additional images from a Wikipedia page"""
        images = []
        
        try:
            # Get page images via API
            images_url = f"https://en.wikipedia.org/api/rest_v1/page/media/{page_id}"
            response = requests.get(images_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('items', []):
                    if item.get('type') == 'image':
                        # Get full image info
                        img_title = item.get('title', '').replace('File:', '')
                        if self.is_food_related_image(img_title):
                            images.append({
                                'url': item.get('original', {}).get('source'),
                                'source': 'wikipedia_gallery',
                                'width': item.get('original', {}).get('width', 0),
                                'height': item.get('original', {}).get('height', 0),
                                'title': img_title,
                                'quality_hint': 'high'
                            })
                            
        except Exception as e:
            print(f"   ⚠️ Wikipedia page images error: {e}")
        
        return images
    
    def scrape_brand_website(self, product_name: str, brand: str) -> List[Dict]:
        """Scrape images from brand's official website"""
        images = []
        
        # Brand-specific scrapers
        brand_lower = brand.lower()
        
        if 'chiquita' in brand_lower:
            images.extend(self.scrape_chiquita_images(product_name))
        elif 'dole' in brand_lower:
            images.extend(self.scrape_dole_images(product_name))
        elif 'organic' in brand_lower or 'bio' in brand_lower:
            images.extend(self.scrape_organic_brand_images(product_name, brand))
        else:
            # Generic brand website scraping
            images.extend(self.scrape_generic_brand_website(product_name, brand))
        
        print(f"   🏷️  Found {len(images)} brand website images")
        return images
    
    def scrape_chiquita_images(self, product_name: str) -> List[Dict]:
        """Scrape high-quality images from Chiquita website"""
        images = []
        
        try:
            # Chiquita product pages
            chiquita_urls = [
                "https://www.chiquita.com/our-products/bananas/",
                "https://www.chiquita.com/recipes/"
            ]
            
            for url in chiquita_urls:
                try:
                    response = requests.get(url, headers=self.headers, timeout=15)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Find product images
                        img_tags = soup.find_all('img', src=True)
                        
                        for img in img_tags:
                            src = img.get('src')
                            if src and self.is_product_image(src, 'banana'):
                                full_url = urljoin(url, src)
                                images.append({
                                    'url': full_url,
                                    'source': 'chiquita_official',
                                    'title': img.get('alt', 'Chiquita Banana'),
                                    'quality_hint': 'brand_official'  # Brand images are usually high quality
                                })
                                
                except Exception as e:
                    print(f"      ⚠️ Chiquita URL failed {url}: {e}")
                    continue
                    
        except Exception as e:
            print(f"   ❌ Chiquita scraping error: {e}")
        
        return images
    
    def scrape_open_food_facts(self, product_name: str, brand: str = None) -> List[Dict]:
        """Scrape images from Open Food Facts database"""
        images = []
        
        try:
            # Search Open Food Facts
            search_query = f"{brand} {product_name}" if brand else product_name
            search_url = f"https://world.openfoodfacts.org/cgi/search.pl"
            
            params = {
                'search_terms': search_query,
                'search_simple': 1,
                'action': 'process',
                'json': 1
            }
            
            response = requests.get(search_url, params=params, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                for product in data.get('products', [])[:3]:  # Top 3 matches
                    # Get product images
                    if 'image_url' in product:
                        images.append({
                            'url': product['image_url'],
                            'source': 'open_food_facts',
                            'title': product.get('product_name', product_name),
                            'quality_hint': 'database'
                        })
                    
                    # Get additional images
                    for key in product.keys():
                        if key.startswith('image_') and key.endswith('_url'):
                            images.append({
                                'url': product[key],
                                'source': 'open_food_facts',
                                'title': f"{product.get('product_name', product_name)} - {key}",
                                'quality_hint': 'database'
                            })
                            
        except Exception as e:
            print(f"   ❌ Open Food Facts error: {e}")
        
        print(f"   🌍 Found {len(images)} Open Food Facts images")
        return images
    
    def scrape_google_images(self, product_name: str, brand: str = None, max_images: int = 5) -> List[Dict]:
        """
        Scrape Google Images (use sparingly to avoid rate limits)
        This is a fallback when other sources don't have enough images
        """
        images = []
        
        # Note: This is a simplified approach. In production, consider using:
        # - Google Custom Search API (requires API key but more reliable)
        # - Alternative image search APIs
        # - Image databases with proper licensing
        
        try:
            search_query = f"{brand} {product_name} high quality" if brand else f"{product_name} fresh fruit high quality"
            
            # This would typically use a proper image search API
            # For now, we'll return a placeholder that indicates more work needed
            print(f"   🔍 Google Images search would find images for: '{search_query}'")
            print(f"      💡 Implement Google Custom Search API for production use")
            
        except Exception as e:
            print(f"   ❌ Google Images error: {e}")
        
        return images
    
    def score_and_rank_images(self, images: List[Dict], product_name: str, brand: str = None) -> List[Dict]:
        """Score and rank images by quality and relevance"""
        
        for img in images:
            score = 0.0
            
            # Source quality scoring
            source_scores = {
                'wikipedia_main': 0.9,
                'wikipedia_gallery': 0.8,
                'chiquita_official': 0.85,
                'brand_official': 0.85,
                'open_food_facts': 0.7,
                'google_images': 0.6
            }
            score += source_scores.get(img.get('source', ''), 0.5)
            
            # Size scoring (prefer larger images)
            width = img.get('width', 0)
            height = img.get('height', 0)
            if width > 800 and height > 600:
                score += 0.3
            elif width > 400 and height > 300:
                score += 0.2
            elif width > 200 and height > 200:
                score += 0.1
            
            # Title/description relevance
            title = (img.get('title', '') + ' ' + img.get('description', '')).lower()
            if product_name.lower() in title:
                score += 0.2
            if brand and brand.lower() in title:
                score += 0.2
            
            # Quality hints
            quality_hints = {
                'high': 0.3,
                'brand_official': 0.25,
                'database': 0.15
            }
            score += quality_hints.get(img.get('quality_hint', ''), 0.0)
            
            img['calculated_score'] = score
        
        # Sort by score (highest first)
        return sorted(images, key=lambda x: x.get('calculated_score', 0), reverse=True)
    
    def download_and_store_images(self, product_id: str, images: List[Dict]) -> List[Dict]:
        """Download and store the best images locally"""
        stored_images = []
        
        for i, img in enumerate(images):
            try:
                # Download image
                response = requests.get(img['url'], headers=self.headers, timeout=20, stream=True)
                
                if response.status_code == 200:
                    # Generate filename
                    url_hash = hashlib.md5(img['url'].encode()).hexdigest()[:8]
                    extension = self.get_image_extension(response.headers.get('content-type', ''))
                    filename = f"{product_id}_{img['source']}_{url_hash}{extension}"
                    
                    local_path = self.images_dir / filename
                    
                    # Save image
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Get image info
                    file_size = local_path.stat().st_size
                    image_hash = self.calculate_image_hash(local_path)
                    
                    # Store in database
                    image_id = str(uuid.uuid4())
                    self.store_scraped_image(
                        image_id, product_id, img['source'], img['url'],
                        str(local_path), image_hash, img.get('width', 0),
                        img.get('height', 0), file_size, img.get('calculated_score', 0.0),
                        i == 0, json.dumps(img)  # First image is primary
                    )
                    
                    stored_images.append({
                        'id': image_id,
                        'local_path': str(local_path),
                        'source': img['source'],
                        'quality_score': img.get('calculated_score', 0.0),
                        'is_primary': i == 0
                    })
                    
                    print(f"      ✅ Downloaded: {filename} ({file_size} bytes)")
                    
                    # Small delay to be respectful
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"      ❌ Download failed for {img['url']}: {e}")
                continue
        
        return stored_images
    
    def store_scraped_image(self, image_id: str, product_id: str, source: str, 
                          original_url: str, local_path: str, image_hash: str,
                          width: int, height: int, file_size: int, quality_score: float,
                          is_primary: bool, metadata: str):
        """Store scraped image info in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO scraped_images (
                    id, product_id, source, original_url, local_path, image_hash,
                    width, height, file_size, quality_score, is_primary, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                image_id, product_id, source, original_url, local_path, image_hash,
                width, height, file_size, quality_score, is_primary, metadata
            ))
    
    def is_food_related_image(self, filename: str) -> bool:
        """Check if image filename suggests it's food-related"""
        food_keywords = [
            'fruit', 'banana', 'apple', 'orange', 'berry', 'food', 'fresh',
            'organic', 'produce', 'nutrition', 'healthy', 'ingredient'
        ]
        filename_lower = filename.lower()
        return any(keyword in filename_lower for keyword in food_keywords)
    
    def is_product_image(self, src: str, product_hint: str) -> bool:
        """Check if image source suggests it's a product image"""
        src_lower = src.lower()
        return (
            product_hint.lower() in src_lower or
            any(keyword in src_lower for keyword in ['product', 'fresh', 'fruit', 'food']) or
            src_lower.endswith(('.jpg', '.jpeg', '.png', '.webp'))
        )
    
    def get_image_extension(self, content_type: str) -> str:
        """Get appropriate file extension from content type"""
        type_map = {
            'image/jpeg': '.jpg',
            'image/png': '.png', 
            'image/webp': '.webp',
            'image/gif': '.gif'
        }
        return type_map.get(content_type, '.jpg')
    
    def calculate_image_hash(self, file_path: Path) -> str:
        """Calculate hash of image file for deduplication"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def scrape_generic_brand_website(self, product_name: str, brand: str) -> List[Dict]:
        """Generic brand website scraping"""
        images = []
        # This would implement generic brand website scraping logic
        # For now, return empty list
        return images
    
    def scrape_dole_images(self, product_name: str) -> List[Dict]:
        """Scrape images from Dole website"""
        images = []
        # This would implement Dole-specific scraping
        return images
    
    def scrape_organic_brand_images(self, product_name: str, brand: str) -> List[Dict]:
        """Scrape images from organic/bio brand websites"""
        images = []
        # This would implement organic brand scraping
        return images

def main():
    """CLI interface for testing image scraper"""
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python image_scraper.py <database_path> <product_name> <brand>")
        return
    
    db_path = sys.argv[1]
    product_name = sys.argv[2]
    brand = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != 'None' else None
    
    scraper = ImageScraper(db_path)
    
    # For demo, use a mock product ID
    product_id = "test-product-id"
    
    images = scraper.scrape_product_images(product_id, product_name, brand)
    
    print(f"\n🎯 Final Results for {product_name}:")
    for img in images:
        print(f"   ✅ {img['source']}: {img['local_path']} (score: {img['quality_score']:.2f})")

if __name__ == "__main__":
    main()