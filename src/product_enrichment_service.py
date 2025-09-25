#!/usr/bin/env python3
"""
Product Enrichment Service
- Google Product API integration
- Web scraping for Flink and other retailer data  
- Product matching and data enrichment
"""

import requests
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, quote_plus
import pandas as pd
from datetime import datetime
import hashlib

# Web scraping imports
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️  Selenium not installed. Run: pip install selenium")

try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False
    print("⚠️  cloudscraper not installed. Run: pip install cloudscraper")


class ProductEnrichmentService:
    """
    Comprehensive product data enrichment service
    """
    
    def __init__(self, output_dir: str = "enriched_products"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.enriched_db_path = self.output_dir / "enriched_products.csv"
        self.scraped_urls_path = self.output_dir / "scraped_urls.json"
        
        # Load existing data
        self.enriched_products = self._load_enriched_products()
        self.scraped_urls = self._load_scraped_urls()
        
        # API configurations
        self.google_api_key = None
        self.google_cx = None  # Custom Search Engine ID
        
    def _load_enriched_products(self) -> pd.DataFrame:
        """Load existing enriched products database"""
        if self.enriched_db_path.exists():
            return pd.read_csv(self.enriched_db_path, dtype=str, na_values=[], keep_default_na=False)
        else:
            return pd.DataFrame(columns=[
                'product_signature', 'original_name', 'brand', 'price', 'quantity',
                'google_title', 'google_description', 'google_image_url', 
                'google_product_url', 'flink_url', 'flink_nutrition_facts',
                'flink_ingredients', 'flink_allergens', 'flink_high_res_images',
                'enrichment_timestamp', 'enrichment_sources'
            ])
    
    def _load_scraped_urls(self) -> Dict[str, Any]:
        """Load cache of scraped URLs to avoid re-scraping"""
        if self.scraped_urls_path.exists():
            with open(self.scraped_urls_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_scraped_urls(self):
        """Save scraped URLs cache"""
        with open(self.scraped_urls_path, 'w') as f:
            json.dump(self.scraped_urls, f, indent=2)
    
    def setup_google_api(self, api_key: str, custom_search_engine_id: str):
        """
        Setup Google Custom Search API
        
        Args:
            api_key: Google API key with Custom Search API enabled
            custom_search_engine_id: Custom Search Engine ID
        """
        self.google_api_key = api_key
        self.google_cx = custom_search_engine_id
        print(f"✅ Google Custom Search API configured")
    
    def search_google_products(self, query: str, num_results: int = 5, search_type: str = 'web') -> List[Dict[str, Any]]:
        """
        Search for products using Google Custom Search API
        
        Args:
            query: Search query (product name + brand)
            num_results: Number of results to return
            search_type: 'web' for nutrition data, 'image' for product images
            
        Returns:
            List of search results with product information
        """
        if not self.google_api_key or not self.google_cx:
            print("❌ Google API not configured. Call setup_google_api() first")
            return []
        
        try:
            # Google Custom Search API endpoint
            url = "https://www.googleapis.com/customsearch/v1"
            
            params = {
                'key': self.google_api_key,
                'cx': self.google_cx,
                'q': query,
                'num': min(num_results, 10),  # Max 10 per request
            }
            
            # Add search type for images only
            if search_type == 'image':
                params['searchType'] = 'image'
                params['fields'] = 'items(title,link,snippet,pagemap)'
            else:
                # Web search - include more fields for better content
                params['fields'] = 'items(title,link,snippet,displayLink,formattedUrl)'
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if 'items' in data:
                for item in data['items']:
                    if search_type == 'image':
                        result = {
                            'title': item.get('title', ''),
                            'snippet': item.get('snippet', ''),
                            'image_url': item.get('link', ''),
                            'page_url': item.get('pagemap', {}).get('cse_image', [{}])[0].get('src', ''),
                            'source': 'google_custom_search'
                        }
                    else:
                        # Web search result
                        result = {
                            'title': item.get('title', ''),
                            'snippet': item.get('snippet', ''),
                            'page_url': item.get('link', ''),  # This is the actual webpage URL
                            'display_link': item.get('displayLink', ''),
                            'formatted_url': item.get('formattedUrl', ''),
                            'source': 'google_custom_search'
                        }
                    results.append(result)
            
            print(f"📊 Found {len(results)} Google results for: {query}")
            return results
            
        except Exception as e:
            print(f"❌ Google search error: {str(e)}")
            return []
    
    def search_flink_product(self, product_name: str, brand: str = '') -> Optional[Dict[str, Any]]:
        """
        Search for specific product on Flink website
        
        Args:
            product_name: Product name to search for
            brand: Brand name (optional)
            
        Returns:
            Product information if found
        """
        search_query = f"{brand} {product_name}".strip()
        
        # Try different search strategies
        search_variations = [
            search_query,
            product_name,
            f"{product_name} {brand}",
            product_name.replace(' ', '+')
        ]
        
        for query in search_variations:
            # Use Google to find Flink product page
            google_query = f"site:flink.de {query}"
            google_results = self.search_google_products(google_query, num_results=3)
            
            for result in google_results:
                if 'flink.de' in result.get('page_url', ''):
                    product_url = result['page_url']
                    
                    # Scrape the Flink product page
                    flink_data = self.scrape_flink_product_page(product_url)
                    if flink_data:
                        return {
                            'flink_url': product_url,
                            'flink_data': flink_data,
                            'source': 'flink_scraping'
                        }
        
        print(f"⚠️  No Flink product found for: {search_query}")
        return None
    
    def scrape_flink_product_page(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape detailed product information from Flink product page
        
        Args:
            url: Flink product page URL
            
        Returns:
            Scraped product data
        """
        # Check cache first
        url_hash = hashlib.md5(url.encode()).hexdigest()
        if url_hash in self.scraped_urls:
            print(f"📋 Using cached data for: {url}")
            return self.scraped_urls[url_hash]
        
        print(f"🕷️  Scraping Flink product: {url}")
        
        # Try cloudscraper first (better bot detection evasion)
        if CLOUDSCRAPER_AVAILABLE:
            try:
                scraper = cloudscraper.create_scraper()
                response = scraper.get(url, timeout=15)
                
                if response.status_code == 200:
                    product_data = self._parse_flink_html(response.text, url)
                    if product_data:
                        # Cache the result
                        self.scraped_urls[url_hash] = product_data
                        self._save_scraped_urls()
                        return product_data
            except Exception as e:
                print(f"⚠️  Cloudscraper failed: {str(e)}")
        
        # Fallback to Selenium for complex pages
        if SELENIUM_AVAILABLE:
            return self._scrape_with_selenium(url, url_hash)
        
        print(f"❌ Unable to scrape: {url}")
        return None
    
    def _scrape_with_selenium(self, url: str, url_hash: str) -> Optional[Dict[str, Any]]:
        """Scrape using Selenium for JavaScript-heavy pages"""
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
            
            with webdriver.Chrome(options=options) as driver:
                driver.get(url)
                
                # Wait for page to load
                wait = WebDriverWait(driver, 10)
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                
                # Take screenshot for manual review
                screenshot_path = self.output_dir / f"screenshot_{url_hash}.png"
                driver.save_screenshot(str(screenshot_path))
                
                # Get page HTML
                html_content = driver.page_source
                
                product_data = self._parse_flink_html(html_content, url)
                if product_data:
                    # Cache the result
                    product_data['screenshot_path'] = str(screenshot_path)
                    self.scraped_urls[url_hash] = product_data
                    self._save_scraped_urls()
                    return product_data
                    
        except Exception as e:
            print(f"⚠️  Selenium scraping failed: {str(e)}")
            
        return None
    
    def _parse_flink_html(self, html: str, url: str) -> Optional[Dict[str, Any]]:
        """
        Parse Flink product page HTML to extract product information
        
        Args:
            html: HTML content
            url: Source URL
            
        Returns:
            Parsed product data
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            print("❌ BeautifulSoup not installed. Run: pip install beautifulsoup4")
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        product_data = {
            'url': url,
            'title': '',
            'price': '',
            'description': '',
            'nutrition_facts': {},
            'ingredients': '',
            'allergens': [],
            'images': [],
            'brand': '',
            'category': '',
            'scraped_at': datetime.now().isoformat()
        }
        
        try:
            # Extract title
            title_selectors = [
                'h1', '.product-title', '[data-testid="product-name"]',
                '.pdp-product-name', '.product-name'
            ]
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    product_data['title'] = title_elem.get_text(strip=True)
                    break
            
            # Extract price
            price_selectors = [
                '.price', '.product-price', '[data-testid="price"]',
                '.current-price', '.price-current'
            ]
            for selector in price_selectors:
                price_elem = soup.select_one(selector)
                if price_elem:
                    product_data['price'] = price_elem.get_text(strip=True)
                    break
            
            # Extract images
            img_elements = soup.find_all('img')
            for img in img_elements:
                src = img.get('src') or img.get('data-src')
                if src and ('product' in src.lower() or 'item' in src.lower()):
                    if src.startswith('//'):
                        src = 'https:' + src
                    elif src.startswith('/'):
                        src = urljoin(url, src)
                    product_data['images'].append(src)
            
            # Extract nutrition information
            nutrition_keywords = ['nutrition', 'nährwert', 'energy', 'calories', 'protein', 'fat', 'carbs']
            for keyword in nutrition_keywords:
                nutrition_elements = soup.find_all(text=re.compile(keyword, re.I))
                for elem in nutrition_elements:
                    parent = elem.parent
                    if parent:
                        nutrition_text = parent.get_text(strip=True)
                        if len(nutrition_text) > 10:  # Meaningful nutrition info
                            product_data['nutrition_facts'][keyword] = nutrition_text
            
            # Extract ingredients
            ingredient_selectors = [
                '[data-testid="ingredients"]', '.ingredients', 
                '.product-ingredients', '.ingredient-list'
            ]
            for selector in ingredient_selectors:
                ingredients_elem = soup.select_one(selector)
                if ingredients_elem:
                    product_data['ingredients'] = ingredients_elem.get_text(strip=True)
                    break
            
            # Look for ingredient keywords in text
            if not product_data['ingredients']:
                ingredient_keywords = ['zutaten', 'ingredients', 'inhaltsstoffe']
                for keyword in ingredient_keywords:
                    ingredient_elements = soup.find_all(text=re.compile(keyword, re.I))
                    for elem in ingredient_elements:
                        parent = elem.parent
                        if parent:
                            text = parent.get_text(strip=True)
                            if len(text) > 20:  # Meaningful ingredients list
                                product_data['ingredients'] = text
                                break
                    if product_data['ingredients']:
                        break
            
            print(f"✅ Parsed Flink product: {product_data['title'][:50]}...")
            return product_data
            
        except Exception as e:
            print(f"❌ HTML parsing error: {str(e)}")
            return None
    
    def enrich_product(self, product_name: str, brand: str = '', price: str = '',
                      quantity: str = '') -> Dict[str, Any]:
        """
        Enrich a single product with external data
        
        Args:
            product_name: Product name from extraction
            brand: Brand name  
            price: Product price
            quantity: Product quantity
            
        Returns:
            Enriched product data
        """
        print(f"🔍 Enriching product: {brand} {product_name}")
        
        # Create product signature for deduplication
        signature = self._create_product_signature(product_name, brand, price, quantity)
        
        enriched_data = {
            'product_signature': signature,
            'original_name': product_name,
            'brand': brand,
            'price': price,
            'quantity': quantity,
            'enrichment_timestamp': datetime.now().isoformat(),
            'enrichment_sources': []
        }
        
        # 1. Search Google for product information
        google_query = f"{brand} {product_name}".strip()
        google_results = self.search_google_products(google_query)
        
        if google_results:
            best_result = google_results[0]  # Take the top result
            enriched_data.update({
                'google_title': best_result.get('title', ''),
                'google_description': best_result.get('snippet', ''),
                'google_image_url': best_result.get('image_url', ''),
                'google_product_url': best_result.get('page_url', '')
            })
            enriched_data['enrichment_sources'].append('google_custom_search')
        
        # 2. Search Flink for detailed nutritional information
        flink_data = self.search_flink_product(product_name, brand)
        
        if flink_data:
            flink_product_data = flink_data['flink_data']
            enriched_data.update({
                'flink_url': flink_data['flink_url'],
                'flink_nutrition_facts': json.dumps(flink_product_data.get('nutrition_facts', {})),
                'flink_ingredients': flink_product_data.get('ingredients', ''),
                'flink_allergens': json.dumps(flink_product_data.get('allergens', [])),
                'flink_high_res_images': json.dumps(flink_product_data.get('images', []))
            })
            enriched_data['enrichment_sources'].append('flink_scraping')
        
        # Convert sources list to string
        enriched_data['enrichment_sources'] = ','.join(enriched_data['enrichment_sources'])
        
        print(f"✅ Enrichment complete with {len(enriched_data['enrichment_sources'].split(','))} sources")
        return enriched_data
    
    def _create_product_signature(self, name: str, brand: str, price: str, quantity: str) -> str:
        """Create unique signature for product deduplication"""
        signature_string = f"{name.lower().strip()}|{brand.lower().strip()}|{price.strip()}|{quantity.strip()}"
        return hashlib.md5(signature_string.encode()).hexdigest()[:16]
    
    def save_enriched_products(self):
        """Save enriched products to CSV"""
        if not self.enriched_products.empty:
            self.enriched_products.to_csv(self.enriched_db_path, index=False)
            print(f"💾 Saved {len(self.enriched_products)} enriched products to {self.enriched_db_path}")
    
    def enrich_products_from_csv(self, csv_path: str, limit: Optional[int] = None):
        """
        Enrich products from existing extraction CSV
        
        Args:
            csv_path: Path to CSV with extracted products
            limit: Maximum number of products to process
        """
        if not Path(csv_path).exists():
            print(f"❌ CSV file not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path, dtype=str, na_values=[], keep_default_na=False)
        print(f"📊 Found {len(df)} products in {csv_path}")
        
        if limit:
            df = df.head(limit)
            print(f"🔢 Processing first {limit} products")
        
        enriched_count = 0
        
        for idx, row in df.iterrows():
            product_name = row.get('Product Name', '')
            brand = row.get('Brand', '')
            price = row.get('Price', '')
            quantity = row.get('Weight/Quantity', '')
            
            if not product_name:
                continue
            
            print(f"\n📦 {idx + 1}/{len(df)}: {product_name}")
            
            try:
                enriched_data = self.enrich_product(product_name, brand, price, quantity)
                
                # Add to enriched products database
                new_row = pd.DataFrame([enriched_data])
                self.enriched_products = pd.concat([self.enriched_products, new_row], ignore_index=True)
                
                enriched_count += 1
                
                # Save periodically
                if enriched_count % 5 == 0:
                    self.save_enriched_products()
                
                # Rate limiting
                time.sleep(1)  # Be nice to APIs
                
            except Exception as e:
                print(f"❌ Error enriching {product_name}: {str(e)}")
                continue
        
        # Final save
        self.save_enriched_products()
        print(f"\n🎉 Enrichment complete! Processed {enriched_count} products")


def setup_google_api_instructions():
    """Print instructions for setting up Google Custom Search API"""
    print("""
🔑 GOOGLE CUSTOM SEARCH API SETUP

1. Go to Google Cloud Console: https://console.cloud.google.com/
2. Create a new project or select existing one
3. Enable Custom Search API:
   - Go to APIs & Services > Library
   - Search for "Custom Search API" 
   - Click Enable

4. Create API Key:
   - Go to APIs & Services > Credentials
   - Click "Create Credentials" > API Key
   - Copy the API key

5. Create Custom Search Engine:
   - Go to: https://cse.google.com/cse/
   - Click "Add" to create new search engine
   - Add sites to search (e.g., flink.de, rewe.de) or search entire web
   - Copy the Search Engine ID

6. Set environment variables:
   export GOOGLE_API_KEY="your_api_key_here"
   export GOOGLE_CX="your_search_engine_id_here"

Usage:
    service = ProductEnrichmentService()
    service.setup_google_api(api_key, cx_id)
    service.enrich_products_from_csv('products.csv')
""")


if __name__ == "__main__":
    setup_google_api_instructions()