"""
Enhanced CSV Export System
Comprehensive product data extraction and CSV generation
"""

import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

class EnhancedCSVExporter:
    """Enhanced CSV exporter with comprehensive product data fields"""
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.products = []
        self.fieldnames = [
            # Core identification
            'record_id',
            'source_screenshot',
            'extracted_image_filename',
            'tile_position',
            'extraction_timestamp',
            
            # Product information
            'product_name',
            'brand',
            'manufacturer',
            'category',
            'subcategory',
            'product_description',
            
            # Pricing data
            'price',
            'price_numeric',
            'currency',
            'price_per_unit',
            'price_per_unit_numeric',
            'discount_percentage',
            'original_price',
            'sale_price',
            
            # Quantity & measurements
            'weight',
            'weight_numeric',
            'weight_unit',
            'quantity',
            'quantity_numeric',
            'quantity_unit',
            'package_size',
            'serving_size',
            
            # Quality indicators
            'organic_certified',
            'bio_label',
            'quality_grade',
            'origin_country',
            'expiry_date',
            
            # Processing metadata
            'extraction_method',
            'ai_confidence_score',
            'consensus_votes',
            'ocr_text_raw',
            'processing_duration_ms',
            'background_removed',
            'button_removed',
            
            # Technical details
            'image_width',
            'image_height',
            'file_size_bytes',
            'color_profile',
            'worker_id',
            
            # Additional data
            'nutritional_info',
            'ingredients_list',
            'allergen_warnings',
            'barcode',
            'product_url',
            'availability_status',
            'stock_level',
            
            # Error handling
            'extraction_errors',
            'data_quality_score',
            'manual_review_required'
        ]
    
    def add_product(self, product_data: Dict[str, Any]):
        """Add a product record with comprehensive data validation"""
        
        # Create standardized product record
        record = {
            'record_id': len(self.products) + 1,
            'extraction_timestamp': datetime.now().isoformat(),
        }
        
        # Fill in all available data
        for field in self.fieldnames:
            record[field] = product_data.get(field, '')
        
        # Process special fields
        record = self._process_pricing(record, product_data)
        record = self._process_measurements(record, product_data)
        record = self._process_quality_indicators(record, product_data)
        record = self._calculate_data_quality_score(record)
        
        self.products.append(record)
    
    def _process_pricing(self, record: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize pricing information"""
        price_str = data.get('price', '')
        
        if price_str:
            # Extract numeric price
            import re
            price_match = re.search(r'(\d+[,.]?\d*)', price_str.replace(',', '.'))
            if price_match:
                try:
                    record['price_numeric'] = float(price_match.group(1))
                    record['currency'] = 'EUR' if '€' in price_str else 'USD' if '$' in price_str else ''
                except ValueError:
                    pass
        
        # Process price per unit
        price_per_unit = data.get('price_per_unit', '')
        if price_per_unit:
            price_unit_match = re.search(r'(\d+[,.]?\d*)', price_per_unit.replace(',', '.'))
            if price_unit_match:
                try:
                    record['price_per_unit_numeric'] = float(price_unit_match.group(1))
                except ValueError:
                    pass
        
        return record
    
    def _process_measurements(self, record: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize weight/quantity information"""
        import re
        
        # Process weight
        weight_str = data.get('weight', '')
        if weight_str:
            weight_match = re.search(r'(\d+[,.]?\d*)\s*(g|kg|ml|l|stk|stück)', weight_str.lower())
            if weight_match:
                try:
                    record['weight_numeric'] = float(weight_match.group(1).replace(',', '.'))
                    record['weight_unit'] = weight_match.group(2)
                except ValueError:
                    pass
        
        # Process quantity
        quantity_str = data.get('quantity', '')
        if quantity_str:
            qty_match = re.search(r'(\d+)', str(quantity_str))
            if qty_match:
                try:
                    record['quantity_numeric'] = int(qty_match.group(1))
                except ValueError:
                    pass
        
        return record
    
    def _process_quality_indicators(self, record: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify quality and certification indicators"""
        product_name = data.get('product_name', '').lower()
        brand = data.get('brand', '').lower()
        
        # Check for organic/bio indicators
        bio_keywords = ['bio', 'organic', 'öko', 'naturland', 'demeter']
        record['organic_certified'] = any(keyword in product_name or keyword in brand 
                                        for keyword in bio_keywords)
        
        record['bio_label'] = 'Bio' in data.get('brand', '') or 'Bio' in data.get('product_name', '')
        
        return record
    
    def _calculate_data_quality_score(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate data completeness and quality score"""
        essential_fields = [
            'product_name', 'price', 'brand', 'category'
        ]
        
        # Count filled essential fields
        filled_essential = sum(1 for field in essential_fields if record.get(field))
        essential_score = (filled_essential / len(essential_fields)) * 100
        
        # Count total filled fields
        total_fields = len([v for v in record.values() if v not in ['', None, 0]])
        completeness_score = (total_fields / len(self.fieldnames)) * 100
        
        # Calculate composite quality score
        record['data_quality_score'] = round((essential_score * 0.7 + completeness_score * 0.3), 2)
        
        # Flag for manual review if quality is low
        record['manual_review_required'] = record['data_quality_score'] < 60
        
        return record
    
    def export_to_csv(self):
        """Export all products to CSV with full field set"""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        with open(self.output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
            
            for product in self.products:
                writer.writerow(product)
        
        print(f"📊 Exported {len(self.products)} products to {self.output_path}")
        return len(self.products)
    
    def export_summary_stats(self, stats_path: str):
        """Export processing statistics and data quality metrics"""
        stats = {
            'total_products': len(self.products),
            'avg_data_quality_score': sum(p.get('data_quality_score', 0) for p in self.products) / max(len(self.products), 1),
            'products_needing_review': sum(1 for p in self.products if p.get('manual_review_required')),
            'organic_products': sum(1 for p in self.products if p.get('organic_certified')),
            'products_with_pricing': sum(1 for p in self.products if p.get('price_numeric')),
            'unique_brands': len(set(p.get('brand', '').lower() for p in self.products if p.get('brand'))),
            'avg_price': sum(p.get('price_numeric', 0) for p in self.products) / max(sum(1 for p in self.products if p.get('price_numeric')), 1),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📈 Exported processing statistics to {stats_path}")
        return stats

# Utility function for converting legacy data
def convert_legacy_products_to_enhanced(legacy_data: List[Dict[str, Any]], 
                                       exporter: EnhancedCSVExporter):
    """Convert existing product data to enhanced format"""
    
    for item in legacy_data:
        enhanced_data = {
            'source_screenshot': item.get('source_screenshot', ''),
            'product_name': item.get('product_name', ''),
            'brand': item.get('brand', ''),
            'price': item.get('price', ''),
            'weight': item.get('weight', ''),
            'quantity': item.get('quantity', ''),
            'category': item.get('category', ''),
            'subcategory': item.get('subcategory', ''),
            'extraction_method': 'Local_Consensus',
            'ai_confidence_score': item.get('ai_confidence', ''),
            'tile_position': item.get('tile_position', ''),
            'background_removed': True,
            'button_removed': True,
            'extracted_image_filename': item.get('image_filename', '')
        }
        
        exporter.add_product(enhanced_data)