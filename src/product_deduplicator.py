#!/usr/bin/env python3
"""
Intelligent Product Deduplication System
Handles product matching, conflict resolution, and data aggregation
"""

import sqlite3
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import pandas as pd

class ConflictType(Enum):
    PRICE_DIFFERENCE = "price_difference"
    WEIGHT_DIFFERENCE = "weight_difference" 
    BRAND_MISMATCH = "brand_mismatch"
    NAME_VARIATION = "name_variation"
    QUANTITY_DIFFERENCE = "quantity_difference"

@dataclass
class ProductConflict:
    conflict_type: ConflictType
    field_name: str
    existing_value: str
    new_value: str
    confidence: float
    requires_user_input: bool

class ProductDeduplicator:
    """
    Intelligent system for product deduplication and conflict resolution
    """
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.setup_database()
    
    def setup_database(self):
        """Create tables for deduplication management"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS product_conflicts (
                    id TEXT PRIMARY KEY,
                    existing_product_id TEXT,
                    new_product_data TEXT,  -- JSON
                    conflict_type TEXT,
                    field_name TEXT,
                    existing_value TEXT,
                    new_value TEXT,
                    status TEXT DEFAULT 'pending',  -- pending, resolved, ignored
                    user_decision TEXT,  -- keep_existing, use_new, merge
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS deduplication_rules (
                    id TEXT PRIMARY KEY,
                    rule_type TEXT,
                    field_name TEXT,
                    threshold_value REAL,
                    auto_resolve BOOLEAN DEFAULT FALSE,
                    resolution_strategy TEXT,  -- merge, keep_newest, keep_highest_confidence
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def process_new_product(self, product_data: Dict) -> Dict:
        """
        Process a new product, checking for duplicates and conflicts
        
        Returns:
        - action: 'create_new', 'update_existing', 'requires_user_input'
        - product_id: existing product ID if found
        - conflicts: list of conflicts detected
        - merged_data: proposed merged data
        """
        
        # Create product signature for matching
        signature = self.create_product_signature(product_data)
        
        # Find potential matches
        matches = self.find_potential_matches(product_data, signature)
        
        if not matches:
            return {
                'action': 'create_new',
                'product_id': None,
                'conflicts': [],
                'merged_data': product_data,
                'confidence': 1.0
            }
        
        # Analyze best match for conflicts
        best_match = matches[0]  # Highest confidence match
        existing_product = self.get_product_by_id(best_match['product_id'])
        
        conflicts = self.detect_conflicts(existing_product, product_data)
        
        if not conflicts:
            # Perfect match - just add new information
            merged_data = self.merge_product_data(existing_product, product_data)
            return {
                'action': 'update_existing',
                'product_id': best_match['product_id'],
                'conflicts': [],
                'merged_data': merged_data,
                'confidence': best_match['confidence']
            }
        
        # Check if conflicts can be auto-resolved
        auto_resolvable = self.check_auto_resolvable(conflicts)
        
        if auto_resolvable:
            merged_data = self.auto_resolve_conflicts(existing_product, product_data, conflicts)
            return {
                'action': 'update_existing',
                'product_id': best_match['product_id'],
                'conflicts': conflicts,
                'merged_data': merged_data,
                'confidence': best_match['confidence'],
                'auto_resolved': True
            }
        
        # Store conflicts for user resolution
        conflict_ids = []
        for conflict in conflicts:
            conflict_id = self.store_conflict(best_match['product_id'], product_data, conflict)
            conflict_ids.append(conflict_id)
        
        return {
            'action': 'requires_user_input',
            'product_id': best_match['product_id'],
            'conflicts': conflicts,
            'conflict_ids': conflict_ids,
            'confidence': best_match['confidence']
        }
    
    def create_product_signature(self, product_data: Dict) -> str:
        """Create unique signature for product matching"""
        # Normalize key fields for matching
        name = str(product_data.get('product_name', '')).lower().strip()
        brand = str(product_data.get('brand', '')).lower().strip()
        weight = str(product_data.get('weight', '')).lower().strip()
        
        # Remove common variations
        name = self.normalize_product_name(name)
        
        signature_text = f"{brand}_{name}_{weight}"
        return hashlib.md5(signature_text.encode()).hexdigest()[:16]
    
    def normalize_product_name(self, name: str) -> str:
        """Normalize product names for better matching"""
        # Remove common variations
        replacements = {
            'bananen': 'banana',
            'äpfel': 'apple', 
            'apfel': 'apple',
            'avocados': 'avocado',
            'erdbeeren': 'strawberry',
            'himbeeren': 'raspberry',
            'heidelbeeren': 'blueberry',
            'brombeeren': 'blackberry',
            'birnen': 'pear',
            'birne': 'pear',
            'ananas': 'pineapple'
        }
        
        normalized = name.lower()
        for german, english in replacements.items():
            if german in normalized:
                normalized = english
                break
                
        # Remove punctuation and extra spaces
        normalized = ''.join(c if c.isalnum() or c.isspace() else '' for c in normalized)
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def find_potential_matches(self, product_data: Dict, signature: str) -> List[Dict]:
        """Find potential matching products"""
        matches = []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Exact signature match
            exact_matches = conn.execute('''
                SELECT *, 1.0 as confidence FROM master_products 
                WHERE product_signature = ?
            ''', (signature,)).fetchall()
            
            matches.extend([dict(row) for row in exact_matches])
            
            # Fuzzy name matching
            if not matches:
                name = self.normalize_product_name(product_data.get('product_name', ''))
                brand = product_data.get('brand', '')
                
                similar_matches = conn.execute('''
                    SELECT *, 0.8 as confidence FROM master_products 
                    WHERE LOWER(canonical_name) LIKE ? 
                    OR LOWER(csv_product_name) LIKE ?
                ''', (f'%{name}%', f'%{name}%')).fetchall()
                
                matches.extend([dict(row) for row in similar_matches])
            
            # Brand + partial name match
            if not matches and product_data.get('brand'):
                brand = product_data.get('brand', '').lower()
                brand_matches = conn.execute('''
                    SELECT *, 0.6 as confidence FROM master_products 
                    WHERE LOWER(csv_brand) LIKE ?
                ''', (f'%{brand}%',)).fetchall()
                
                matches.extend([dict(row) for row in brand_matches])
        
        # Sort by confidence descending
        matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)
        return matches[:5]  # Return top 5 matches
    
    def get_product_by_id(self, product_id: str) -> Dict:
        """Get product by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            product = conn.execute('''
                SELECT * FROM master_products WHERE id = ?
            ''', (product_id,)).fetchone()
            
            return dict(product) if product else {}
    
    def detect_conflicts(self, existing: Dict, new: Dict) -> List[ProductConflict]:
        """Detect conflicts between existing and new product data"""
        conflicts = []
        
        # Price conflicts
        existing_price = self.parse_price(existing.get('csv_price', ''))
        new_price = self.parse_price(new.get('price', ''))
        
        if existing_price and new_price and abs(existing_price - new_price) > 0.10:
            conflicts.append(ProductConflict(
                conflict_type=ConflictType.PRICE_DIFFERENCE,
                field_name='price',
                existing_value=existing.get('csv_price', ''),
                new_value=new.get('price', ''),
                confidence=0.8,
                requires_user_input=abs(existing_price - new_price) > 1.0
            ))
        
        # Brand conflicts
        existing_brand = existing.get('csv_brand', '').lower().strip()
        new_brand = new.get('brand', '').lower().strip()
        
        if existing_brand and new_brand and existing_brand != new_brand:
            conflicts.append(ProductConflict(
                conflict_type=ConflictType.BRAND_MISMATCH,
                field_name='brand',
                existing_value=existing.get('csv_brand', ''),
                new_value=new.get('brand', ''),
                confidence=0.9,
                requires_user_input=True
            ))
        
        # Weight conflicts
        if existing.get('csv_weight') and new.get('weight'):
            if existing['csv_weight'] != new['weight']:
                conflicts.append(ProductConflict(
                    conflict_type=ConflictType.WEIGHT_DIFFERENCE,
                    field_name='weight',
                    existing_value=existing.get('csv_weight', ''),
                    new_value=new.get('weight', ''),
                    confidence=0.7,
                    requires_user_input=False
                ))
        
        # Name variations
        existing_name = existing.get('csv_product_name', '').lower()
        new_name = new.get('product_name', '').lower()
        
        if existing_name and new_name and existing_name != new_name:
            # Check if they're similar enough to be the same product
            similarity = self.calculate_name_similarity(existing_name, new_name)
            if similarity < 0.8:  # Not similar enough
                conflicts.append(ProductConflict(
                    conflict_type=ConflictType.NAME_VARIATION,
                    field_name='product_name',
                    existing_value=existing.get('csv_product_name', ''),
                    new_value=new.get('product_name', ''),
                    confidence=similarity,
                    requires_user_input=similarity < 0.6
                ))
        
        return conflicts
    
    def parse_price(self, price_str: str) -> Optional[float]:
        """Parse price string to float"""
        if not price_str:
            return None
        
        try:
            # Remove currency symbols and convert comma to dot
            price_clean = str(price_str).replace('€', '').replace('$', '').replace(',', '.').strip()
            return float(price_clean)
        except (ValueError, TypeError):
            return None
    
    def calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two product names"""
        # Simple Levenshtein distance approach
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        max_len = max(len(name1), len(name2))
        if max_len == 0:
            return 1.0
        
        distance = levenshtein_distance(name1, name2)
        return 1 - (distance / max_len)
    
    def check_auto_resolvable(self, conflicts: List[ProductConflict]) -> bool:
        """Check if all conflicts can be auto-resolved"""
        return all(not conflict.requires_user_input for conflict in conflicts)
    
    def auto_resolve_conflicts(self, existing: Dict, new: Dict, conflicts: List[ProductConflict]) -> Dict:
        """Auto-resolve conflicts using predefined rules"""
        merged = existing.copy()
        
        for conflict in conflicts:
            if conflict.conflict_type == ConflictType.WEIGHT_DIFFERENCE:
                # Keep both weight variations
                weights = [existing.get('csv_weight'), new.get('weight')]
                merged['weight_variations'] = list(filter(None, weights))
                
            elif conflict.conflict_type == ConflictType.PRICE_DIFFERENCE:
                # Keep the more recent price, store historical prices
                merged['csv_price'] = new.get('price', existing.get('csv_price'))
                merged['price_history'] = merged.get('price_history', [])
                merged['price_history'].append({
                    'price': existing.get('csv_price'),
                    'date': existing.get('created_at')
                })
        
        # Add any new information from new product
        for key, value in new.items():
            if key not in merged or not merged[key]:
                merged[key] = value
        
        return merged
    
    def merge_product_data(self, existing: Dict, new: Dict) -> Dict:
        """Merge new product data with existing product (no conflicts)"""
        merged = existing.copy()
        
        # Add any new fields that don't exist or are empty in existing
        for key, value in new.items():
            if value and (key not in merged or not merged[key]):
                merged[key] = value
        
        # Special handling for arrays/lists
        if new.get('images'):
            existing_images = merged.get('images', [])
            merged['images'] = existing_images + [img for img in new['images'] if img not in existing_images]
        
        # Update enrichment score if new data provides more information
        if new.get('enrichment_data'):
            merged['enrichment_score'] = max(
                merged.get('enrichment_score', 0),
                self.calculate_enrichment_score(merged)
            )
        
        # Update last modified timestamp
        merged['last_updated'] = datetime.now().isoformat()
        
        return merged
    
    def store_conflict(self, product_id: str, new_data: Dict, conflict: ProductConflict) -> str:
        """Store conflict for user resolution"""
        import uuid
        conflict_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO product_conflicts (
                    id, existing_product_id, new_product_data, conflict_type,
                    field_name, existing_value, new_value, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                conflict_id,
                product_id,
                json.dumps(new_data),
                conflict.conflict_type.value,
                conflict.field_name,
                conflict.existing_value,
                conflict.new_value,
                'pending'
            ))
        
        return conflict_id
    
    def get_pending_conflicts(self) -> List[Dict]:
        """Get all pending conflicts requiring user input"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            conflicts = conn.execute('''
                SELECT c.*, p.csv_product_name, p.csv_brand
                FROM product_conflicts c
                LEFT JOIN master_products p ON c.existing_product_id = p.id
                WHERE c.status = 'pending'
                ORDER BY c.created_at DESC
            ''').fetchall()
            
            return [dict(row) for row in conflicts]
    
    def resolve_conflict(self, conflict_id: str, decision: str, merged_data: Dict = None):
        """Resolve a conflict with user decision"""
        with sqlite3.connect(self.db_path) as conn:
            # Update conflict status
            conn.execute('''
                UPDATE product_conflicts 
                SET status = 'resolved', user_decision = ?, resolved_at = ?
                WHERE id = ?
            ''', (decision, datetime.now().isoformat(), conflict_id))
            
            # Apply the resolution if merged data provided
            if merged_data:
                conflict = conn.execute('''
                    SELECT existing_product_id FROM product_conflicts WHERE id = ?
                ''', (conflict_id,)).fetchone()
                
                if conflict:
                    self.update_product(conflict[0], merged_data)
    
    def update_product(self, product_id: str, updated_data: Dict):
        """Update existing product with merged data"""
        with sqlite3.connect(self.db_path) as conn:
            # Build update query dynamically based on available data
            update_fields = []
            update_values = []
            
            field_mapping = {
                'csv_price': 'price',
                'csv_brand': 'brand', 
                'csv_product_name': 'product_name',
                'csv_weight': 'weight',
                'csv_quantity': 'quantity',
                'csv_extracted_text': 'extracted_text'
            }
            
            for db_field, data_field in field_mapping.items():
                if data_field in updated_data:
                    update_fields.append(f"{db_field} = ?")
                    update_values.append(updated_data[data_field])
            
            if update_fields:
                query = f"UPDATE master_products SET {', '.join(update_fields)} WHERE id = ?"
                update_values.append(product_id)
                conn.execute(query, update_values)
    
    def calculate_enrichment_score(self, product_data: Dict) -> int:
        """Calculate enrichment score based on available data"""
        score = 0
        max_score = 100
        
        # Basic information (40 points)
        if product_data.get('csv_product_name'):
            score += 10
        if product_data.get('csv_brand'):
            score += 10
        if product_data.get('csv_price'):
            score += 10
        if product_data.get('csv_weight'):
            score += 10
        
        # Images (20 points)
        if product_data.get('images'):
            score += min(20, len(product_data['images']) * 5)
        
        # Nutritional data (30 points)
        if product_data.get('nutrition_data'):
            score += 30
        elif product_data.get('canonical_name'):
            score += 10  # At least we know the food type
        
        # External enrichment (10 points)
        if product_data.get('external_data_sources'):
            score += 10
        
        return min(score, max_score)

def main():
    """CLI interface for testing deduplication"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python product_deduplicator.py <database_path>")
        return
    
    db_path = sys.argv[1]
    deduplicator = ProductDeduplicator(db_path)
    
    # Show pending conflicts
    conflicts = deduplicator.get_pending_conflicts()
    
    if conflicts:
        print(f"📋 {len(conflicts)} pending conflicts:")
        for conflict in conflicts:
            print(f"   🔍 {conflict['field_name']}: {conflict['existing_value']} vs {conflict['new_value']}")
            print(f"      Product: {conflict['csv_product_name']} ({conflict['csv_brand']})")
    else:
        print("✅ No pending conflicts")

if __name__ == "__main__":
    main()