#!/usr/bin/env python3
"""
Vision LLM Food Analyzer
Analyzes images to identify food items using vision language models
"""

import base64
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
import sqlite3
import uuid
from datetime import datetime

class VisionFoodAnalyzer:
    """
    Analyzes food images using vision LLMs to extract product information
    """
    
    def __init__(self, db_path: str, api_key: str = None):
        self.db_path = Path(db_path)
        self.api_key = api_key
        self.setup_database()
    
    def setup_database(self):
        """Create tables for vision analysis results"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS vision_analysis (
                    id TEXT PRIMARY KEY,
                    image_path TEXT NOT NULL,
                    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Vision LLM Results
                    food_items_detected TEXT,  -- JSON array of detected items
                    confidence_score REAL,
                    single_item BOOLEAN,
                    multiple_items BOOLEAN,
                    
                    -- Brand/Product Detection
                    brand_detected TEXT,
                    product_name_detected TEXT,
                    logo_detected BOOLEAN,
                    text_extracted TEXT,
                    
                    -- Matching Results
                    existing_matches TEXT,  -- JSON array of matching product IDs
                    new_product_created BOOLEAN,
                    
                    -- Raw Analysis
                    raw_llm_response TEXT
                )
            ''')
    
    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64 for API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def analyze_food_image(self, image_path: str) -> Dict:
        """
        Analyze an image to identify food items
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # For now, simulate the vision analysis with a mock response
        # In production, this would call OpenAI Vision API or similar
        mock_analysis = self._mock_vision_analysis(image_path)
        
        # Store analysis result
        analysis_id = str(uuid.uuid4())
        self._store_analysis_result(analysis_id, image_path, mock_analysis)
        
        # Find matching products in database
        matches = self._find_matching_products(mock_analysis)
        
        # Update analysis with matches
        self._update_matches(analysis_id, matches)
        
        return {
            'analysis_id': analysis_id,
            'image_path': str(image_path),
            'food_items': mock_analysis['food_items'],
            'confidence': mock_analysis['confidence'],
            'single_item': mock_analysis['single_item'],
            'brand_detected': mock_analysis.get('brand'),
            'existing_matches': matches,
            'recommendations': self._generate_recommendations(mock_analysis, matches)
        }
    
    def _mock_vision_analysis(self, image_path: Path) -> Dict:
        """
        Mock vision analysis - replace with real LLM API call
        
        In production, this would send the image to:
        - OpenAI Vision API (GPT-4V)
        - Google Gemini Vision
        - Anthropic Claude 3 Vision
        """
        
        # Determine food type based on filename (for demo purposes)
        filename = image_path.name.lower()
        
        if 'banana' in filename or 'chiquita' in filename:
            return {
                'food_items': [
                    {
                        'name': 'Cavendish Banana',
                        'variety': 'Cavendish',
                        'quantity': 'bunch',
                        'quantity_count': 6,
                        'ripeness': 'ripe',
                        'size_estimate': 'medium',
                        'confidence': 0.95,
                        'description': 'Fresh yellow Cavendish bananas in natural bunch formation'
                    }
                ],
                'single_item': True,
                'multiple_items': False,
                'brand': 'Chiquita' if 'chiquita' in filename else 'Generic',
                'brand_confidence': 0.90 if 'chiquita' in filename else 0.30,
                'all_text_extracted': ['Chiquita', 'Premium', 'Ecuador'] if 'chiquita' in filename else ['Bananas', 'Product of Ecuador'],
                'packaging_type': 'loose produce',
                'certifications': ['Rainforest Alliance'] if 'chiquita' in filename else [],
                'pricing_info': {
                    'price_visible': False,
                    'price_text': None,
                    'currency': None,
                    'unit': 'per bunch',
                    'store_brand': None,
                    'barcode_visible': False,
                    'plu_code': '4011'
                },
                'origin_info': {
                    'country_of_origin': 'Ecuador',
                    'region': 'Costa Rica' if 'chiquita' in filename else 'Ecuador',
                    'harvest_date': None,
                    'best_before': None,
                    'growing_method': 'conventional'
                },
                'nutritional_visible': {
                    'nutrition_panel': False,
                    'ingredients_list': False,
                    'calories_shown': False,
                    'health_claims': ['potassium rich', 'natural energy']
                },
                'context': {
                    'setting': 'grocery store produce section',
                    'lighting': 'natural bright',
                    'other_items_visible': [],
                    'seasonal_context': 'available year-round',
                    'cultural_context': 'global staple fruit'
                },
                'overall_confidence': 0.95,
                'analysis_completeness': 'comprehensive',
                'recommended_database_category': 'fruit',
                'recommended_subcategory': 'tropical fruit',
                'estimated_retail_value': '$1.50 per bunch',
                'raw_response': 'I can see fresh yellow Cavendish bananas in a natural bunch formation. They appear to be perfectly ripe with the characteristic yellow color and slight green at the stem.'
            }
        
        elif 'apple' in filename:
            variety = 'Pink Lady' if 'pink' in filename else 'Gala' if 'gala' in filename else 'Red Delicious'
            return {
                'food_items': [
                    {
                        'name': f'{variety} Apple',
                        'variety': variety,
                        'quantity': 'single',
                        'quantity_count': 1,
                        'ripeness': 'fresh',
                        'size_estimate': 'medium-large',
                        'confidence': 0.92,
                        'description': f'Fresh {variety} apple with characteristic coloring and shape'
                    }
                ],
                'single_item': True,
                'multiple_items': False,
                'brand': 'Stemilt' if 'organic' in filename else None,
                'brand_confidence': 0.75 if 'organic' in filename else 0.20,
                'all_text_extracted': ['Premium', 'Washington State'] if 'organic' in filename else [variety],
                'packaging_type': 'loose produce',
                'certifications': ['USDA Organic'] if 'organic' in filename else [],
                'pricing_info': {
                    'price_visible': False,
                    'price_text': None,
                    'currency': 'USD',
                    'unit': 'per pound',
                    'store_brand': None,
                    'barcode_visible': False,
                    'plu_code': '4130' if variety == 'Pink Lady' else '4174'
                },
                'origin_info': {
                    'country_of_origin': 'USA',
                    'region': 'Washington State',
                    'harvest_date': None,
                    'best_before': None,
                    'growing_method': 'organic' if 'organic' in filename else 'conventional'
                },
                'nutritional_visible': {
                    'nutrition_panel': False,
                    'ingredients_list': False,
                    'calories_shown': False,
                    'health_claims': ['high fiber', 'vitamin C rich']
                },
                'context': {
                    'setting': 'grocery store produce section',
                    'lighting': 'bright retail',
                    'other_items_visible': [],
                    'seasonal_context': 'fall harvest peak',
                    'cultural_context': 'american favorite'
                },
                'overall_confidence': 0.92,
                'analysis_completeness': 'comprehensive',
                'recommended_database_category': 'fruit',
                'recommended_subcategory': 'apple',
                'estimated_retail_value': '$2.49 per pound',
                'raw_response': f'I can see a fresh {variety} apple with excellent color and appearance, indicating high quality and freshness.'
            }
        
        elif 'avocado' in filename:
            return {
                'food_items': [
                    {
                        'name': 'avocado',
                        'variety': 'hass avocado',
                        'quantity': 'single',
                        'confidence': 0.88,
                        'description': 'Ripe Hass avocado'
                    }
                ],
                'confidence': 0.88,
                'single_item': True,
                'brand': 'Bio' if 'bio' in filename else None,
                'text_extracted': 'Bio Avocado' if 'bio' in filename else '',
                'raw_response': 'I can see a ripe Hass avocado in this image.'
            }
        
        else:
            # Default analysis for unknown items
            return {
                'food_items': [
                    {
                        'name': 'unknown food item',
                        'quantity': 'unknown',
                        'confidence': 0.3,
                        'description': 'Unable to identify specific food item'
                    }
                ],
                'confidence': 0.3,
                'single_item': True,
                'brand': None,
                'text_extracted': '',
                'raw_response': 'I can see some food items but cannot identify them clearly.'
            }
    
    def _real_vision_analysis(self, image_path: Path) -> Dict:
        """
        Real vision analysis using OpenAI Vision API
        (Commented out for now - requires API key and credits)
        """
        if not self.api_key:
            raise ValueError("API key required for real vision analysis")
        
        base64_image = self.encode_image(image_path)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
                            Analyze this food image COMPREHENSIVELY and extract ALL possible information. Provide a detailed JSON response with:

                            ## FOOD IDENTIFICATION
                            1. What exact food items do you see? (be specific: "Pink Lady apples" not just "apples")
                            2. How many of each item? (single, bunch, package, etc.)
                            3. What's the ripeness/quality state? (ripe, overripe, fresh, etc.)
                            4. Any visible variety/type? (Granny Smith, Roma tomatoes, etc.)

                            ## BRAND & PACKAGING
                            5. Any brand names, logos, or company identifiers visible?
                            6. Extract ALL visible text from packaging, labels, stickers
                            7. What type of packaging? (plastic bag, cardboard box, loose, etc.)
                            8. Any certifications? (organic, fair trade, non-GMO, etc.)

                            ## PRICING & RETAIL INFO
                            9. Any price tags, barcodes, or retail stickers visible?
                            10. Store chain identifiers? (Whole Foods, Walmart, etc.)
                            11. Product codes, PLU numbers, or SKUs?
                            12. Weight or quantity information? (1kg, 6-pack, etc.)

                            ## NUTRITIONAL CLUES
                            13. Can you see nutrition facts panels?
                            14. Ingredient lists visible?
                            15. Calorie information shown?
                            16. Any health claims? ("low fat", "high protein", etc.)

                            ## ORIGIN & QUALITY
                            17. Country of origin visible? ("Product of USA", etc.)
                            18. Harvest date, best before, expiration dates?
                            19. Quality indicators? (grade A, premium, etc.)
                            20. Growing method clues? (greenhouse, field grown, hydroponic, etc.)

                            ## VISUAL CHARACTERISTICS
                            21. Size estimation compared to common objects?
                            22. Color variations or patterns?
                            23. Shape irregularities or unique features?
                            24. Surface texture (smooth, rough, waxy, etc.)

                            ## CONTEXT CLUES
                            25. What type of setting? (grocery store, kitchen, market, etc.)
                            26. Other items visible that provide context?
                            27. Seasonal indicators?
                            28. Cultural or regional food indicators?

                            Return comprehensive JSON format:
                            {
                                "food_items": [
                                    {
                                        "name": "Pink Lady Apple",
                                        "variety": "Pink Lady",
                                        "quantity": "single",
                                        "quantity_count": 1,
                                        "ripeness": "fresh",
                                        "size_estimate": "medium",
                                        "confidence": 0.95,
                                        "description": "Fresh Pink Lady apple with characteristic pink-red coloring"
                                    }
                                ],
                                "single_item": true,
                                "multiple_items": false,
                                
                                "brand": "Organic Girl",
                                "brand_confidence": 0.85,
                                "all_text_extracted": ["Organic Girl", "Premium Quality", "Product of USA", "$2.99/lb"],
                                "packaging_type": "plastic clamshell",
                                "certifications": ["USDA Organic", "Non-GMO"],
                                
                                "pricing_info": {
                                    "price_visible": true,
                                    "price_text": "$2.99/lb",
                                    "currency": "USD",
                                    "unit": "per pound",
                                    "store_brand": null,
                                    "barcode_visible": false,
                                    "plu_code": "4130"
                                },
                                
                                "origin_info": {
                                    "country_of_origin": "USA",
                                    "region": "Washington State",
                                    "harvest_date": null,
                                    "best_before": null,
                                    "growing_method": "conventional"
                                },
                                
                                "nutritional_visible": {
                                    "nutrition_panel": false,
                                    "ingredients_list": false,
                                    "calories_shown": false,
                                    "health_claims": ["heart healthy", "high fiber"]
                                },
                                
                                "context": {
                                    "setting": "grocery store produce section",
                                    "lighting": "bright fluorescent",
                                    "other_items_visible": ["bananas", "oranges"],
                                    "seasonal_context": "available year-round",
                                    "cultural_context": "western supermarket"
                                },
                                
                                "overall_confidence": 0.92,
                                "analysis_completeness": "comprehensive",
                                "recommended_database_category": "fruit",
                                "recommended_subcategory": "apple",
                                "estimated_retail_value": "$2.99 per pound"
                            }
                            
                            BE THOROUGH - extract every visible detail that could help with product database enrichment!
                            """
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            try:
                # Parse JSON response
                analysis = json.loads(content)
                analysis['raw_response'] = content
                return analysis
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    'food_items': [{'name': 'unknown', 'confidence': 0.1}],
                    'single_item': True,
                    'brand': None,
                    'text_extracted': '',
                    'confidence': 0.1,
                    'raw_response': content
                }
        else:
            raise Exception(f"Vision API error: {response.status_code} - {response.text}")
    
    def _store_analysis_result(self, analysis_id: str, image_path: Path, analysis: Dict):
        """Store vision analysis result in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO vision_analysis (
                    id, image_path, food_items_detected, confidence_score,
                    single_item, multiple_items, brand_detected, 
                    product_name_detected, text_extracted, raw_llm_response
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_id,
                str(image_path),
                json.dumps(analysis['food_items']),
                analysis['confidence'],
                analysis['single_item'],
                not analysis['single_item'],
                analysis.get('brand'),
                analysis['food_items'][0]['name'] if analysis['food_items'] else None,
                analysis.get('text_extracted', ''),
                analysis.get('raw_response', '')
            ))
    
    def _find_matching_products(self, analysis: Dict) -> List[Dict]:
        """Find existing products that match the detected food items"""
        matches = []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            for food_item in analysis['food_items']:
                food_name = food_item['name'].lower()
                brand = (analysis.get('brand') or '').lower()
                
                # Search for matching products
                query_conditions = []
                query_params = []
                
                # Match by canonical name
                query_conditions.append("LOWER(canonical_name) LIKE ?")
                query_params.append(f"%{food_name}%")
                
                # If brand detected, prioritize brand matches
                if brand:
                    query_conditions.append("LOWER(csv_brand) LIKE ?")
                    query_params.append(f"%{brand}%")
                    where_clause = " AND ".join(query_conditions)
                else:
                    where_clause = query_conditions[0]
                
                products = conn.execute(f'''
                    SELECT * FROM master_products 
                    WHERE {where_clause}
                    ORDER BY enrichment_score DESC
                    LIMIT 5
                ''', query_params).fetchall()
                
                for product in products:
                    matches.append({
                        'product_id': product['id'],
                        'product_name': product['csv_product_name'],
                        'brand': product['csv_brand'],
                        'canonical_name': product['canonical_name'],
                        'enrichment_score': product['enrichment_score'],
                        'match_type': 'exact_brand' if brand and brand in (product['csv_brand'] or '').lower() else 'name_only'
                    })
        
        return matches
    
    def _update_matches(self, analysis_id: str, matches: List[Dict]):
        """Update analysis record with matching results"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE vision_analysis 
                SET existing_matches = ?, new_product_created = ?
                WHERE id = ?
            ''', (
                json.dumps(matches),
                len(matches) == 0,  # Create new product if no matches
                analysis_id
            ))
    
    def _generate_recommendations(self, analysis: Dict, matches: List[Dict]) -> List[str]:
        """Generate recommendations based on analysis and matches"""
        recommendations = []
        
        if not matches:
            recommendations.append("No existing products found - this appears to be a new item")
            recommendations.append("Consider adding this as a new product to your database")
        
        elif len(matches) == 1:
            match = matches[0]
            if match['match_type'] == 'exact_brand':
                recommendations.append(f"Perfect match found: {match['product_name']} by {match['brand']}")
                recommendations.append("Consider adding this image to the existing product profile")
            else:
                recommendations.append(f"Similar product found: {match['product_name']}")
                recommendations.append("Verify if this is the same product or a different variant")
        
        else:
            recommendations.append(f"Multiple similar products found ({len(matches)})")
            recommendations.append("Review matches to determine if this is an existing product or new variant")
        
        # Brand-specific recommendations
        brand = analysis.get('brand')
        if brand:
            recommendations.append(f"Brand '{brand}' detected - search for other {brand} products")
            recommendations.append(f"Consider fetching high-quality images from {brand}'s website")
        
        return recommendations

def main():
    """CLI interface for testing vision analysis"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python vision_analyzer.py <database_path> <image_path>")
        return
    
    db_path = sys.argv[1]
    image_path = sys.argv[2]
    
    analyzer = VisionFoodAnalyzer(db_path)
    result = analyzer.analyze_food_image(image_path)
    
    print("🔍 Vision Analysis Results:")
    print(f"   📸 Image: {result['image_path']}")
    print(f"   🥗 Food Items: {len(result['food_items'])}")
    
    for item in result['food_items']:
        print(f"      • {item['name']} (confidence: {item['confidence']:.2f})")
    
    if result['brand_detected']:
        print(f"   🏷️  Brand: {result['brand_detected']}")
    
    print(f"   🎯 Confidence: {result['confidence']:.2f}")
    print(f"   🔗 Existing Matches: {len(result['existing_matches'])}")
    
    for match in result['existing_matches']:
        print(f"      → {match['product_name']} ({match['match_type']})")
    
    print("\n💡 Recommendations:")
    for rec in result['recommendations']:
        print(f"   • {rec}")

if __name__ == "__main__":
    main()