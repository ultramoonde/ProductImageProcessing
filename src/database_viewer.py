#!/usr/bin/env python3
"""
Master Product Database Viewer
Web interface for viewing and managing enriched product database
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename

# Configure Flask to find templates in parent directory
app = Flask(__name__, template_folder='../templates')

class DatabaseViewer:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
    
    def get_all_products(self, limit: int = 100, offset: int = 0):
        """Get all products with enrichment data"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            products = conn.execute('''
                SELECT 
                    id, product_signature, csv_brand, csv_product_name, csv_price, csv_weight,
                    canonical_name, canonical_brand, canonical_category, enrichment_score, 
                    normalized_price_eur, ai_brand, ai_category, ai_confidence, created_at
                FROM master_products 
                ORDER BY enrichment_score DESC, created_at DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset)).fetchall()
            
            return [dict(row) for row in products]
    
    def get_product_details(self, product_id: str):
        """Get detailed view of single product with all sources"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get product
            product = conn.execute('''
                SELECT * FROM master_products WHERE id = ?
            ''', (product_id,)).fetchone()
            
            if not product:
                return None
            
            # Get original uploaded images (primary thumbnails)
            images = conn.execute('''
                SELECT * FROM product_images WHERE product_id = ? ORDER BY is_primary DESC, created_at ASC
            ''', (product_id,)).fetchall()
            
            # Get scraped images (high-quality)
            scraped_images = conn.execute('''
                SELECT * FROM scraped_images WHERE product_id = ? ORDER BY quality_score DESC
            ''', (product_id,)).fetchall()
            
            # Get nutrition facts
            nutrition = conn.execute('''
                SELECT * FROM nutrition_facts WHERE product_id = ? ORDER BY created_at DESC LIMIT 1
            ''', (product_id,)).fetchone()
            
            return {
                'product': dict(product),
                'sources': [],  # No enrichment sources in simple schema
                'images': [dict(row) for row in images],
                'scraped_images': [dict(row) for row in scraped_images],
                'nutrition': dict(nutrition) if nutrition else None
            }
    
    def search_products(self, query: str):
        """Search products by name, brand, or ingredients"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            products = conn.execute('''
                SELECT 
                    id, csv_product_name, csv_brand, canonical_name, canonical_brand,
                    enrichment_score, canonical_category, ai_brand, ai_category
                FROM master_products 
                WHERE csv_product_name LIKE ? OR csv_brand LIKE ? OR canonical_name LIKE ?
                ORDER BY enrichment_score DESC
                LIMIT 20
            ''', (f'%{query}%', f'%{query}%', f'%{query}%')).fetchall()
            
            return [dict(row) for row in products]
    
    def get_database_stats(self):
        """Get comprehensive database statistics"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Basic counts
            total_products = conn.execute('SELECT COUNT(*) FROM master_products').fetchone()[0]
            enriched_products = conn.execute('SELECT COUNT(*) FROM master_products WHERE enrichment_score > 0').fetchone()[0]
            
            # Average price
            avg_price = conn.execute('''
                SELECT AVG(normalized_price_eur) 
                FROM master_products 
                WHERE normalized_price_eur > 0
            ''').fetchone()[0]
            
            # Images count
            images_found = conn.execute('SELECT COUNT(*) FROM product_images WHERE file_exists = 1').fetchone()[0]
            
            # Category breakdown
            categories = conn.execute('''
                SELECT canonical_category, COUNT(*) as count
                FROM master_products 
                WHERE canonical_category IS NOT NULL
                GROUP BY canonical_category
                ORDER BY count DESC
            ''').fetchall()
            
            # Brand breakdown
            brands = conn.execute('''
                SELECT csv_brand, COUNT(*) as count
                FROM master_products 
                WHERE csv_brand IS NOT NULL AND csv_brand != ''
                GROUP BY csv_brand
                ORDER BY count DESC
                LIMIT 5
            ''').fetchall()
            
            return {
                'total_products': total_products,
                'enriched_products': enriched_products,
                'enrichment_rate': (enriched_products/total_products*100) if total_products > 0 else 0,
                'average_price': f"€{avg_price:.2f}" if avg_price else "€0.00",
                'images_found': images_found,
                'categories': dict(categories),
                'top_brands': [f"{brand} ({count})" for brand, count in brands],
                'sources': []  # No external sources in simple schema
            }

# Global viewer instance
db_viewer = None

@app.route('/')
def index():
    """Database dashboard"""
    stats = db_viewer.get_database_stats()
    return render_template('database_dashboard.html', stats=stats)

@app.route('/api/products')
def api_products():
    """API endpoint for products list"""
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))
    
    products = db_viewer.get_all_products(limit, offset)
    return jsonify({'products': products})

@app.route('/api/products/<product_id>')
def api_product_details(product_id):
    """API endpoint for single product details"""
    details = db_viewer.get_product_details(product_id)
    
    if not details:
        return jsonify({'error': 'Product not found'}), 404
    
    return jsonify(details)

@app.route('/api/search')
def api_search():
    """API endpoint for product search"""
    query = request.args.get('q', '')
    
    if not query:
        return jsonify({'products': []})
    
    products = db_viewer.search_products(query)
    return jsonify({'products': products})

@app.route('/api/stats')
def api_stats():
    """API endpoint for database statistics"""
    stats = db_viewer.get_database_stats()
    return jsonify(stats)

@app.route('/products')
def products_page():
    """Products listing page"""
    return render_template('products_list.html')

@app.route('/products/<product_id>')
def product_detail_page(product_id):
    """Individual product detail page"""
    return render_template('product_detail.html', product_id=product_id)

@app.route('/test')
def test_dashboard():
    """Test dashboard for different input scenarios"""
    return render_template('test_dashboard.html')

@app.route('/uploads/<path:filename>')
def serve_uploaded_image(filename):
    """Serve uploaded images"""
    try:
        upload_dir = Path('uploads')
        return send_from_directory(str(upload_dir), filename)
    except Exception as e:
        return jsonify({'error': f'Image not found: {str(e)}'}), 404

@app.route('/api/enrich/<product_id>', methods=['POST'])
def api_enrich_product(product_id):
    """API endpoint to comprehensively enrich a single product with ALL available data"""
    try:
        # Import the comprehensive enrichment orchestrator
        from enrichment_orchestrator import EnrichmentOrchestrator
        
        # Configuration with real Nutritionix API credentials
        from nutritionix_config import get_nutritionix_config
        nutritionix_config = get_nutritionix_config()
        
        config = {
            'nutritionix_api_key': nutritionix_config['api_key'],
            'nutritionix_app_id': nutritionix_config['app_id'],
            'usda_api_key': request.json.get('usda_api_key') if request.is_json and request.json else None,
            'google_api_key': request.json.get('google_api_key') if request.is_json and request.json else None,
        }
        
        orchestrator = EnrichmentOrchestrator(str(db_viewer.db_path), config)
        result = orchestrator.enrich_product_comprehensive(product_id)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'Comprehensive enrichment failed: {str(e)}'
        }), 500

@app.route('/api/enrich/all', methods=['POST'])
def api_enrich_all_products():
    """API endpoint to comprehensively enrich all products"""
    try:
        from enrichment_orchestrator import EnrichmentOrchestrator
        
        # Configuration with real Nutritionix API credentials
        from nutritionix_config import get_nutritionix_config
        nutritionix_config = get_nutritionix_config()
        
        config = {
            'nutritionix_api_key': nutritionix_config['api_key'],
            'nutritionix_app_id': nutritionix_config['app_id'],
            'usda_api_key': request.json.get('usda_api_key') if request.is_json and request.json else None,
            'google_api_key': request.json.get('google_api_key') if request.is_json and request.json else None,
        }
        
        orchestrator = EnrichmentOrchestrator(str(db_viewer.db_path), config)
        batch_size = int(request.json.get('batch_size', 3)) if request.is_json else 3
        delay = float(request.json.get('delay', 2.0)) if request.is_json else 2.0
        
        results = orchestrator.enrich_all_products(batch_size=batch_size, delay=delay)
        
        return jsonify({
            'status': 'success',
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error', 
            'error': str(e)
        }), 500

@app.route('/api/enrich/demo', methods=['POST'])
def api_enrich_demo():
    """Demo enrichment with mock data (no API keys required)"""
    try:
        # Get product ID from request
        product_id = request.json.get('product_id') if request.is_json else None
        if not product_id:
            return jsonify({'error': 'Product ID required'}), 400
        
        # Run image scraping only (no API keys needed)
        from simple_image_enricher import SimpleImageEnricher
        
        enricher = SimpleImageEnricher(str(db_viewer.db_path))
        
        # Get product info
        with sqlite3.connect(db_viewer.db_path) as conn:
            conn.row_factory = sqlite3.Row
            product = conn.execute('SELECT * FROM master_products WHERE id = ?', (product_id,)).fetchone()
            
            if not product:
                return jsonify({'error': 'Product not found'}), 404
        
        # Run image enrichment
        images = enricher.enrich_product_images(
            product_id, 
            product['csv_product_name'], 
            product['csv_brand']
        )
        
        # Update enrichment score
        new_score = min(product['enrichment_score'] + 25, 100)  # Add 25 points for images
        
        conn.execute('''
            UPDATE master_products 
            SET enrichment_score = ?, last_enriched = ?
            WHERE id = ?
        ''', (new_score, datetime.now().isoformat(), product_id))
        
        return jsonify({
            'status': 'success',
            'images_found': len(images),
            'new_enrichment_score': new_score,
            'message': f'Found {len(images)} high-quality images and updated enrichment score'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'Demo enrichment failed: {str(e)}'
        }), 500

# Test API Endpoints for Different Input Scenarios
@app.route('/api/test/image-only', methods=['POST'])
def api_test_image_only():
    """Test endpoint for image-only input"""
    try:
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'status': 'error', 'error': 'No image file selected'}), 400
        
        from test_api_handler import TestAPIHandler
        handler = TestAPIHandler(str(db_viewer.db_path))
        result = handler.handle_image_only(image_file)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/test/text-only', methods=['POST'])
def api_test_text_only():
    """Test endpoint for text-only input"""
    try:
        data = request.get_json()
        if not data or not data.get('description'):
            return jsonify({'status': 'error', 'error': 'No description provided'}), 400
        
        from test_api_handler import TestAPIHandler
        handler = TestAPIHandler(str(db_viewer.db_path))
        result = handler.handle_text_only(data['description'])
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/test/text-plus-image', methods=['POST'])
def api_test_text_plus_image():
    """Test endpoint for text + image input"""
    try:
        product_name = request.form.get('product_name', '').strip()
        image_file = request.files.get('image')
        
        if not product_name and not image_file:
            return jsonify({'status': 'error', 'error': 'Either product name or image is required'}), 400
        
        from test_api_handler import TestAPIHandler
        handler = TestAPIHandler(str(db_viewer.db_path))
        result = handler.handle_text_plus_image(product_name, image_file)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/test/new-product', methods=['POST'])
def api_test_new_product():
    """Test endpoint for new product creation"""
    try:
        name = request.form.get('name', '').strip()
        if not name:
            return jsonify({'status': 'error', 'error': 'Product name is required'}), 400
        
        brand = request.form.get('brand', '').strip() or None
        price = request.form.get('price', '').strip() or None
        weight = request.form.get('weight', '').strip() or None
        image_file = request.files.get('image')
        
        from test_api_handler import TestAPIHandler
        handler = TestAPIHandler(str(db_viewer.db_path))
        result = handler.handle_new_product(name, brand, price, weight, image_file)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python database_viewer.py <database_path>")
        sys.exit(1)
    
    db_path = sys.argv[1]
    db_viewer = DatabaseViewer(db_path)
    
    print(f"🌐 Starting database viewer for: {db_path}")
    print("📊 Access dashboard at: http://localhost:5002")
    
    app.run(debug=True, host='0.0.0.0', port=5002)