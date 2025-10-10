"""
Product-Category Validation System

This module provides intelligent validation and correction of category assignments
based on product content analysis. It can detect when LLM-detected categories
don't logically match the actual products and suggest corrections.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CategoryMatch:
    """Represents a category match with confidence score"""
    main_category: str
    subcategory: str
    confidence: float
    reasoning: str

class ProductCategoryValidator:
    """
    Validates and corrects category assignments based on product analysis.
    Uses keyword matching, product type inference, and logical validation.
    """

    def __init__(self):
        # Define product-to-category mapping patterns
        self.category_patterns = {
            "Aufschnitt & Brotaufstriche": {
                "Fruchtaufstriche": [
                    # Jam/preserve keywords
                    r"\b(marmelade|konfitüre|gelee|fruchtaufstrich)\b",
                    r"\b(samt|extra|weniger zucker|bio)\b.*\b(himbeere|erdbeere|aprikose|kirsche|pflaume|orange|zitrone)\b",
                    r"\b(himbeere|erdbeere|aprikose|kirsche|pflaume|orange|zitrone)\b.*\b(aufstrich|jam|preserve)\b",
                    r"\bschwartau\b",  # Brand known for jams
                    r"\bschwarzer?\w*\b.*\b(frucht|beere)\b",
                    # Product name patterns
                    r"samt.*\b(himbeere|erdbeere|aprikose)\b",
                    r"extra.*weniger zucker.*\b(himbeere|erdbeere|aprikose|kirsche)\b",
                ],
                "Nuss- & Schokoaufstriche": [
                    r"\b(nutella|nuss|haselnuss|erdnuss|mandel)\b.*\b(aufstrich|creme)\b",
                    r"\b(schoko|schokoladen|kakao)\b.*\b(aufstrich|creme)\b",
                    r"\b(nougat|praline)\b",
                ],
                "Wurst & Fleischaufstriche": [
                    r"\b(leberwurst|mettwurst|teewurst|salami)\b",
                    r"\b(fleisch|wurst)\b.*\b(aufstrich|paste)\b",
                ]
            },
            "Käse": {
                "Käse": [
                    r"\b(käse|cheese|gouda|cheddar|camembert|brie|mozzarella)\b",
                    r"\b(frischkäse|schmelzkäse|hartkäse|weichkäse)\b",
                ]
            },
            "Fleisch & Fisch": {
                "Fleisch": [
                    r"\b(fleisch|meat|rind|schwein|huhn|hähnchen|pute|kalb)\b",
                    r"\b(steak|schnitzel|hackfleisch|bratwurst)\b",
                ],
                "Fisch": [
                    r"\b(fisch|fish|lachs|thunfisch|forelle|kabeljau|hering)\b",
                    r"\b(salmon|tuna|cod|herring)\b",
                ]
            },
            "Obst & Gemüse": {
                "Obst": [
                    r"\b(apfel|apple|banane|orange|birne|traube|kiwi|mango)\b",
                    r"\b(beeren|erdbeere|himbeere|blaubeere|brombeere)\b",
                ],
                "Gemüse": [
                    r"\b(gemüse|vegetable|tomate|gurke|paprika|zwiebel|karotte)\b",
                    r"\b(salat|spinat|brokkoli|blumenkohl|zucchini)\b",
                ]
            }
        }

        # Brand-to-category mappings for additional context
        self.brand_mappings = {
            "schwartau": ("Aufschnitt & Brotaufstriche", "Fruchtaufstriche"),
            "schwarzwald": ("Aufschnitt & Brotaufstriche", "Fruchtaufstriche"),
            "schwartzau": ("Aufschnitt & Brotaufstriche", "Fruchtaufstriche"),  # Common OCR error
            "nutella": ("Aufschnitt & Brotaufstriche", "Nuss- & Schokoaufstriche"),
        }

    def validate_categories_with_products(self, current_categories: Dict, products: List[Dict]) -> Dict:
        """
        Main validation method that analyzes products and corrects category assignments.

        Args:
            current_categories: Current category detection result
            products: List of detected products with names, brands, etc.

        Returns:
            Corrected category data with validation info
        """
        print("🔍 PRODUCT-CATEGORY VALIDATION")
        print("=" * 50)

        # Analyze products to infer correct categories
        product_analysis = self._analyze_products(products)

        # Find the most confident category match
        best_match = self._find_best_category_match(product_analysis)

        if best_match:
            print(f"📊 Product analysis suggests: {best_match.main_category} → {best_match.subcategory}")
            print(f"🎯 Confidence: {best_match.confidence:.2f}")
            print(f"💡 Reasoning: {best_match.reasoning}")

            # Compare with current detection
            current_main = current_categories.get('main_category', 'Unknown')
            current_sub = current_categories.get('active_subcategory', 'Unknown')

            print(f"🔄 Current detection: {current_main} → {current_sub}")

            # Decide whether to override
            should_override = self._should_override_detection(
                current_categories, best_match, product_analysis
            )

            if should_override:
                print("✅ OVERRIDING category detection based on product analysis")

                # Create corrected category data
                corrected_categories = {
                    'main_category': best_match.main_category,
                    'active_subcategory': best_match.subcategory,
                    'available_subcategories': self._get_available_subcategories(best_match.main_category),
                    'confidence': min(best_match.confidence, 0.9),  # Cap confidence
                    'method': 'product_validated_correction',
                    'original_detection': current_categories,
                    'validation_info': {
                        'product_match_confidence': best_match.confidence,
                        'reasoning': best_match.reasoning,
                        'analyzed_products': len(products),
                        'matching_products': len([p for p in product_analysis if p['category_matches']])
                    },
                    'all_detected_categories': current_categories.get('all_detected_categories', [])
                }

                return corrected_categories
            else:
                print("ℹ️ Keeping original detection (insufficient confidence for override)")

        # Return original categories with validation info
        current_categories['validation_info'] = {
            'validated': True,
            'product_analysis_performed': True,
            'override_confidence': best_match.confidence if best_match else 0.0,
            'analyzed_products': len(products)
        }

        return current_categories

    def _analyze_products(self, products: List[Dict]) -> List[Dict]:
        """Analyze each product to determine its likely category"""
        analysis_results = []

        for i, product in enumerate(products):
            print(f"   📦 Analyzing product {i+1}: {product.get('product_name', 'Unknown')}")

            product_analysis = {
                'product': product,
                'category_matches': [],
                'brand_match': None,
                'confidence': 0.0
            }

            # Get product text for analysis
            product_text = self._extract_product_text(product)
            print(f"      📝 Analysis text: '{product_text}'")

            # Check brand mappings first
            brand = product.get('brand', '').lower()
            if brand in self.brand_mappings:
                main_cat, sub_cat = self.brand_mappings[brand]
                product_analysis['brand_match'] = (main_cat, sub_cat)
                product_analysis['confidence'] += 0.3
                print(f"      🏷️ Brand match: {brand} → {main_cat}/{sub_cat}")

            # Check category patterns
            for main_category, subcategories in self.category_patterns.items():
                for subcategory, patterns in subcategories.items():
                    for pattern in patterns:
                        if re.search(pattern, product_text, re.IGNORECASE):
                            match_strength = self._calculate_match_strength(pattern, product_text)
                            product_analysis['category_matches'].append({
                                'main_category': main_category,
                                'subcategory': subcategory,
                                'pattern': pattern,
                                'strength': match_strength
                            })
                            print(f"      ✅ Pattern match: '{pattern}' → {main_category}/{subcategory} (strength: {match_strength:.2f})")

            analysis_results.append(product_analysis)

        return analysis_results

    def _extract_product_text(self, product: Dict) -> str:
        """Extract all text fields from product for analysis"""
        text_parts = []

        # Include all text fields
        for field in ['product_name', 'brand', 'description']:
            value = product.get(field, '')
            if value and value.strip():
                text_parts.append(value.strip())

        return ' '.join(text_parts)

    def _calculate_match_strength(self, pattern: str, text: str) -> float:
        """Calculate how strong a pattern match is"""
        matches = re.findall(pattern, text, re.IGNORECASE)
        if not matches:
            return 0.0

        # Base strength
        strength = 0.5

        # Bonus for multiple matches
        if len(matches) > 1:
            strength += 0.2

        # Bonus for specific product terms
        if re.search(r'\b(samt|extra|weniger zucker)\b', text, re.IGNORECASE):
            strength += 0.2

        # Bonus for fruit names in jam context
        if re.search(r'\b(himbeere|erdbeere|aprikose|kirsche)\b', text, re.IGNORECASE):
            strength += 0.2

        return min(strength, 1.0)

    def _find_best_category_match(self, product_analysis: List[Dict]) -> Optional[CategoryMatch]:
        """Find the most confident category match across all products"""
        category_scores = {}

        for analysis in product_analysis:
            # Score from brand matches
            if analysis['brand_match']:
                main_cat, sub_cat = analysis['brand_match']
                key = (main_cat, sub_cat)
                category_scores[key] = category_scores.get(key, 0) + 0.4

            # Score from pattern matches
            for match in analysis['category_matches']:
                key = (match['main_category'], match['subcategory'])
                category_scores[key] = category_scores.get(key, 0) + match['strength']

        if not category_scores:
            return None

        # Find best match
        best_key = max(category_scores.keys(), key=lambda k: category_scores[k])
        best_score = category_scores[best_key]

        # Create reasoning
        matching_products = len([a for a in product_analysis if a['category_matches'] or a['brand_match']])
        total_products = len(product_analysis)

        reasoning = f"Found {matching_products}/{total_products} products matching {best_key[0]}→{best_key[1]} patterns"

        return CategoryMatch(
            main_category=best_key[0],
            subcategory=best_key[1],
            confidence=min(best_score / total_products, 1.0),
            reasoning=reasoning
        )

    def _should_override_detection(self, current: Dict, suggested: CategoryMatch, analysis: List[Dict]) -> bool:
        """Decide whether to override current detection with suggested category"""

        # Override if confidence is high
        if suggested.confidence >= 0.7:
            return True

        # Override if current detection has low confidence and suggestion is reasonable
        current_confidence = current.get('confidence', 0.8)
        if current_confidence < 0.5 and suggested.confidence >= 0.4:
            return True

        # Override if there's clear product-category mismatch
        current_main = current.get('main_category', '').lower()
        if current_main == 'käse' and suggested.main_category == 'Aufschnitt & Brotaufstriche':
            # Cheese vs jams/spreads - clear mismatch for jam products
            matching_products = len([a for a in analysis if a['category_matches']])
            if matching_products >= 2:  # At least 2 products match the suggestion
                return True

        return False

    def _get_available_subcategories(self, main_category: str) -> List[str]:
        """Get available subcategories for a main category"""
        if main_category in self.category_patterns:
            return list(self.category_patterns[main_category].keys())
        return []