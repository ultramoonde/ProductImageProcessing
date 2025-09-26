#!/usr/bin/env python3
"""
LocalConsensusAnalyzer - Integrated working consensus system
Combines the working fixed system with original method signatures for compatibility.
Supports both UI (category) analysis and product analysis with 3-model consensus.
"""

import cv2
import numpy as np
import json
import requests
import base64
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
import easyocr
import re

class LocalConsensusAnalyzer:
    """
    Integrated consensus analyzer supporting both UI and product analysis.
    Uses 3-model consensus with Ollama vision models.
    """

    def __init__(self, use_api_fallback: bool = False):
        """Initialize the consensus analyzer."""
        # Note: use_api_fallback parameter kept for compatibility but ignored
        # as per user request: "I only want the OCR processing to be happening via the consensus system"
        self.reader = easyocr.Reader(['en', 'de'])

        # Working models with proper weights
        self.models = [
            {"name": "llama3.2-vision:11b", "weight": 1.0},
            {"name": "minicpm-v:latest", "weight": 1.2},
            {"name": "moondream:latest", "weight": 0.8}
        ]

        print("🔧 INTEGRATED CONSENSUS SYSTEM INITIALIZED")
        print(f"📋 Using models: {[m['name'] for m in self.models]}")

    def _extract_ocr_text(self, image: np.ndarray) -> str:
        """Extract OCR text from image using EasyOCR."""
        try:
            results = self.reader.readtext(image, paragraph=True)
            if results:
                # Handle both 2-element and 3-element tuples
                texts = []
                for result in results:
                    if len(result) == 3:
                        bbox, text, conf = result
                        if conf > 0.3:
                            texts.append(text)
                    elif len(result) == 2:
                        text, conf = result
                        if conf > 0.3:
                            texts.append(text)
                return ' | '.join(texts) if texts else ""
            return ""
        except Exception as e:
            print(f"❌ OCR Error: {e}")
            return ""

    def _create_local_analysis_prompt(self, ocr_text: str, analysis_mode: str, custom_prompt: str = None) -> str:
        """Create analysis prompt based on mode (ui/product/coordinate_mapping)."""
        if custom_prompt:
            return custom_prompt
        elif analysis_mode == "coordinate_mapping":
            # This should not happen as coordinate mapping always uses custom prompt
            return custom_prompt or "Analyze image for coordinate information."
        elif analysis_mode == "ui":
            # SIMPLIFIED UI analysis focused on visual hierarchy
            return f"""Analyze this Flink grocery app header to identify categories.

SIMPLE RULES:
1. TOP ROW with colored/pink background = MAIN CATEGORY
2. BOTTOM ROW with bold/centered text = ACTIVE SUBCATEGORY

VISUAL LAYOUT:
- Row 1: "Categories" title
- Row 2: [Category with pink background] ← MAIN CATEGORY
- Row 3: [Bold text] [Normal text] [Normal text] ← ACTIVE = bold text

FALLBACK: If you can't determine bold text, return your best guess and mark confidence as low.

Return JSON exactly like this:
{{
  "main_category": "text from pink background row",
  "active_subcategory": "bold/centered text from bottom row",
  "available_subcategories": ["all", "visible", "subcategories"],
  "confidence": "high/medium/low",
  "fallback_reason": "explanation if confidence is low"
}}

EXAMPLES:
- Pink background "Obst & Gemüse" + bold "Bananen" → main: "Obst & Gemüse", active: "Bananen"
- Pink background "Konserven, Instantgerichte & Backen" + bold "Back- & Dessertmischungen" → main: "Konserven, Instantgerichte & Backen", active: "Back- & Dessertmischungen"

Only return valid JSON. Focus on the visual styling: pink background = main, bold text = active subcategory."""

        else:
            # Enhanced product analysis for comprehensive extraction
            return f"""Analyze this German grocery product image and extract ALL information with precise formatting.

OCR detected text: "{ocr_text}"

=== EXTRACTION REQUIREMENTS ===

1. BRAND IDENTIFICATION (CRITICAL):
   - Look for manufacturer/company names (usually prominent text or first word)
   - Common German grocery brands: Aurora, Berief, Alnatura, Bio Company, Demeter, Rapunzel, Edeka, Rewe, etc.
   - Brand vs Product distinction examples:
     * "Aurora Sonnenstern-Grieß" → brand: "Aurora", product: "Sonnenstern-Grieß"
     * "Berief Bio Hafer" → brand: "Berief", product: "Bio Hafer"
     * "Chiquita Bananen" → brand: "Chiquita", product: "Bananen"
   - Generic terms are NOT brands: "Bio", "Organic", "Classic", "Type 405" are descriptors
   - If no clear manufacturer visible, use empty string ""

2. STRICT FORMATTING RULES:
   - price: Exact format "X,XX €" (German format with comma)
   - weight: NUMERIC VALUE ONLY (no units) → "0.5", "500", "1"
   - unit: STANDARDIZED UNITS → "kg", "g", "l", "ml", "Stk"
   - quantity: COUNT ONLY → "1", "2", "5" (for multipacks, pieces)

3. UNIT STANDARDIZATION:
   - Weight products: Use "kg" for ≥1kg, "g" for <1kg
   - Liquid products: Use "l" for ≥1l, "ml" for <1l
   - Count products: Use "Stk" for pieces/items
   - NEVER mix units in weight field: "0,5kg" → weight: "0.5", unit: "kg"

4. PRICE CALCULATION (MANDATORY):
   - Calculate price per standard unit based on product type:
   - Weight products → price_per_kg (convert g to kg: 500g = 0.5kg)
   - Liquid products → price_per_liter (convert ml to l: 500ml = 0.5l)
   - Count products → price_per_piece
   - Multipack calculation: "2x1l for 4,16€" → 4.16÷2 = 2.08€/l

=== EXAMPLES ===

German flour product:
{{"price": "1,69 €", "product_name": "Sonnenstern-Grieß Weichweizen", "brand": "Aurora", "weight": "0.5", "unit": "kg", "quantity": "", "price_per_kg": "3.38", "price_per_piece": "", "price_per_liter": ""}}

Liquid multipack:
{{"price": "4,16 €", "product_name": "Bio Hafer Barista", "brand": "Berief", "weight": "1", "unit": "l", "quantity": "2", "price_per_kg": "", "price_per_piece": "", "price_per_liter": "2.08"}}

Count product:
{{"price": "2,99 €", "product_name": "Bananen", "brand": "Chiquita", "weight": "", "unit": "Stk", "quantity": "5", "price_per_kg": "", "price_per_piece": "0.60", "price_per_liter": ""}}

Small weight product:
{{"price": "0,89 €", "product_name": "Backpulver", "brand": "Dr. Oetker", "weight": "16", "unit": "g", "quantity": "", "price_per_kg": "55.63", "price_per_piece": "", "price_per_liter": ""}}

=== MANDATORY JSON FORMAT ===
{{"price": "", "product_name": "", "brand": "", "weight": "", "unit": "", "quantity": "", "price_per_kg": "", "price_per_piece": "", "price_per_liter": ""}}

Analyze both the main product image and text areas thoroughly. Extract brand names carefully. Calculate prices accurately. Return ONLY the JSON, no other text."""

    def _calculate_cost_metrics(self, product_data: Dict) -> Dict:
        """
        Calculate cost per kg and cost per piece from extracted product data.
        Uses mutually exclusive logic:
        - If weight (kg/g) is provided -> calculate cost per kg only
        - If no weight but quantity (Stk./pieces) is provided -> calculate cost per piece only
        """
        price_str = product_data.get("price", "")
        weight_str = product_data.get("weight", "")
        quantity_str = product_data.get("quantity", "")
        description = product_data.get("product_name", "")

        cost_per_kg = ""
        cost_per_piece = ""

        try:
            # Extract numeric price value
            if price_str:
                # Handle German format "1,49 €" or "1.49 €"
                price_numeric = 0.0
                import re
                price_match = re.search(r'(\d+[,.]?\d*)', price_str.replace(',', '.'))
                if price_match:
                    price_numeric = float(price_match.group(1))

                    # SMART DETECTION: Determine if this is weight-based or piece-based product
                    is_weight_based = False
                    is_piece_based = False

                    # Check for weight indicators in description or weight field
                    weight_indicators = ['kg', 'g ', 'gram', 'ml', 'liter', 'l ']
                    piece_indicators = ['stk', 'stück', ' x ', 'pieces', 'pack']

                    # Priority 1: Check if explicit weight is provided
                    if weight_str:
                        weight_match = re.search(r'(\d+[,.]?\d*)', weight_str.replace(',', '.'))
                        if weight_match and any(indicator in weight_str.lower() for indicator in weight_indicators):
                            is_weight_based = True

                    # Priority 2: Check description for weight/piece indicators
                    if not is_weight_based and description:
                        description_lower = description.lower()
                        if any(indicator in description_lower for indicator in weight_indicators):
                            is_weight_based = True
                        elif any(indicator in description_lower for indicator in piece_indicators):
                            is_piece_based = True

                    # Priority 3: If no clear indication, check quantity field
                    if not is_weight_based and not is_piece_based and quantity_str:
                        quantity_lower = quantity_str.lower()
                        if any(indicator in quantity_lower for indicator in piece_indicators):
                            is_piece_based = True

                    # CALCULATE BASED ON PRODUCT TYPE (MUTUALLY EXCLUSIVE)
                    if is_weight_based and weight_str:
                        # WEIGHT-BASED PRODUCT: Calculate cost per kg only
                        weight_match = re.search(r'(\d+[,.]?\d*)', weight_str.replace(',', '.'))
                        if weight_match:
                            weight_numeric = float(weight_match.group(1))

                            # Convert to kg if needed
                            if 'g' in weight_str.lower() and 'kg' not in weight_str.lower():
                                weight_numeric = weight_numeric / 1000  # Convert grams to kg
                            elif 'ml' in weight_str.lower():
                                weight_numeric = weight_numeric / 1000  # Convert ml to l (approximate for liquids)

                            if weight_numeric > 0:
                                cost_per_kg = f"{(price_numeric / weight_numeric):.2f} €/kg"

                    elif is_piece_based and quantity_str:
                        # PIECE-BASED PRODUCT: Calculate cost per piece only
                        quantity_match = re.search(r'(\d+)', quantity_str)
                        if quantity_match:
                            quantity_numeric = int(quantity_match.group(1))
                            if quantity_numeric > 0:
                                cost_per_piece = f"{(price_numeric / quantity_numeric):.2f} €/Stk"

                    # Fallback: If no clear type detected, use quantity if available
                    elif not is_weight_based and quantity_str:
                        quantity_match = re.search(r'(\d+)', quantity_str)
                        if quantity_match:
                            quantity_numeric = int(quantity_match.group(1))
                            if quantity_numeric > 0:
                                cost_per_piece = f"{(price_numeric / quantity_numeric):.2f} €/Stk"

        except (ValueError, ZeroDivisionError) as e:
            print(f"      ⚠️  Cost calculation error: {e}")

        return {
            "cost_per_kg": cost_per_kg,
            "cost_per_piece": cost_per_piece
        }

    async def _query_single_local_model(self, model: Dict, image_base64: str, text_base64: str, ocr_text: str, analysis_mode: str, custom_prompt: str = None) -> Dict:
        """Query single Ollama model with proper error handling."""
        model_name = model["name"]
        start_time = time.time()

        try:
            # Create appropriate prompt
            prompt = self._create_local_analysis_prompt(ocr_text, analysis_mode, custom_prompt)

            # Prepare payload for vision models
            # For product analysis, prioritize text region for better text reading
            if analysis_mode == "product" and text_base64 != image_base64:
                # Use text region for product text analysis
                analysis_image = text_base64
            else:
                # Use main image for UI analysis or when no separate text region
                analysis_image = image_base64

            payload = {
                "model": model_name,
                "prompt": prompt,
                "images": [analysis_image],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 2048  # Increase token limit for complete JSON responses
                }
            }

            # Make synchronous request in executor
            def make_request():
                try:
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json=payload,
                        timeout=60  # Longer timeout for vision models
                    )
                    return response
                except requests.exceptions.RequestException:
                    return None

            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                future = loop.run_in_executor(executor, make_request)
                response = await future

            processing_time = time.time() - start_time

            if response is None:
                return {
                    "status": "connection_error",
                    "raw_response": "Connection failed",
                    "error": "Could not connect to Ollama",
                    "processing_time": processing_time
                }

            if response.status_code == 200:
                result = response.json()
                raw_content = result.get('response', '').strip()
# Debug logging removed

                # Try to parse JSON
                try:
                    if raw_content.startswith('{') and raw_content.endswith('}'):
                        parsed_json = json.loads(raw_content)
                        return {
                            "status": "success",
                            "raw_response": raw_content,
                            "parsed_data": parsed_json,
                            "processing_time": processing_time,
                            "parse_status": "direct_json"
                        }
                    else:
                        # Extract JSON from text with improved regex for nested JSON
                        # Try to find complete JSON objects with proper nesting
                        json_matches = []

                        # Try multiple JSON extraction patterns
                        patterns = [
                            r'\{(?:[^{}]|{[^{}]*})*\}',  # Handles one level of nesting
                            r'\{[^}]*\}',  # Simple pattern as fallback
                            r'\{[^$]*',    # Captures truncated JSON (from { to end of string)
                        ]

                        for pattern in patterns:
                            matches = re.findall(pattern, raw_content, re.DOTALL)
                            for match in matches:
                                try:
                                    parsed_json = json.loads(match)
                                    print(f"   ✅ DEBUG: Successfully extracted JSON: {list(parsed_json.keys())}")
                                    return {
                                        "status": "extracted_json",
                                        "raw_response": raw_content,
                                        "parsed_data": parsed_json,
                                        "processing_time": processing_time,
                                        "parse_status": "extracted_json"
                                    }
                                except json.JSONDecodeError as e:
                                    print(f"   ❌ DEBUG: JSON decode failed for match: '{match[:100]}...' Error: {e}")
                                    # Try to repair truncated JSON
                                    if match.startswith('{') and not match.endswith('}'):
                                        try:
                                            # Handle common truncation patterns
                                            repaired = match.rstrip(',').rstrip()

                                            # Fix incomplete field names like "available_s..."
                                            if '"available_s' in repaired and not repaired.endswith('"'):
                                                # Find the position and clean up
                                                pos = repaired.rfind('"available_s')
                                                if pos != -1:
                                                    repaired = repaired[:pos].rstrip(',').rstrip()

                                            # Fix incomplete field names like "cate..."
                                            elif '"cate' in repaired and not repaired.endswith('"'):
                                                pos = repaired.rfind('"cate')
                                                if pos != -1:
                                                    repaired = repaired[:pos].rstrip(',').rstrip()

                                            repaired += '}'
                                            parsed_json = json.loads(repaired)
                                            print(f"   🔧 DEBUG: Repaired truncated JSON: {list(parsed_json.keys())}")
                                            return {
                                                "status": "extracted_json",
                                                "raw_response": raw_content,
                                                "parsed_data": parsed_json,
                                                "processing_time": processing_time,
                                                "parse_status": "repaired_json"
                                            }
                                        except json.JSONDecodeError:
                                            print(f"   ❌ DEBUG: JSON repair failed")
                                            continue
                                    continue

                        print(f"   🔍 DEBUG: No JSON extracted from content: '{raw_content[:500]}...'")
                        return {
                            "status": "no_json",
                            "raw_response": raw_content,
                            "error": "No valid JSON found in response",
                            "processing_time": processing_time,
                            "parse_status": "no_json"
                        }

                except json.JSONDecodeError as e:
                    print(f"   🔍 DEBUG JSON Parse Error for content: '{raw_content[:200]}...'")
                    return {
                        "status": "json_error",
                        "raw_response": raw_content,
                        "error": f"JSON parse error: {e}",
                        "processing_time": processing_time,
                        "parse_status": "json_error"
                    }
            else:
                error_text = response.text if hasattr(response, 'text') else "Unknown error"
                return {
                    "status": "http_error",
                    "raw_response": error_text,
                    "error": f"HTTP {response.status_code}: {error_text}",
                    "processing_time": processing_time
                }

        except Exception as e:
            processing_time = time.time() - start_time
            print(f"   ⚠️ Exception in {model_name}: {str(e)}")
            return {
                "status": "exception",
                "raw_response": f"Exception occurred: {str(e)}",
                "error": str(e),
                "processing_time": processing_time
            }

    async def analyze_product_with_consensus(self, tile_image: np.ndarray, text_region_image: np.ndarray, analysis_mode: str = "product", custom_prompt: str = None) -> Dict:
        """
        Main consensus analysis method - maintains original signature for compatibility.
        Supports both product and UI analysis modes.

        Args:
            tile_image: Primary image for analysis
            text_region_image: Text region image (can be same as tile_image)
            analysis_mode: "product" for product analysis, "ui" for UI/category analysis

        Returns:
            Dict with consensus results
        """
        print(f"\n🧠 INTEGRATED CONSENSUS ANALYSIS - MODE: {analysis_mode.upper()}")
        print("=" * 60)

        # Extract OCR text from primary image
        ocr_text = self._extract_ocr_text(tile_image)
        print(f"📝 OCR: '{ocr_text}'")

        # Convert images to base64
        _, buffer = cv2.imencode('.png', tile_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Use text_region_image for text analysis (for better OCR/reading)
        if text_region_image is not None and text_region_image.shape != tile_image.shape:
            _, text_buffer = cv2.imencode('.png', text_region_image)
            text_base64 = base64.b64encode(text_buffer).decode('utf-8')
            print(f"🖼️  Tile image: {len(image_base64)} chars, Text region: {len(text_base64)} chars")
        else:
            text_base64 = image_base64  # Fallback to same image if text region not available
            print(f"🖼️  Using single image: {len(image_base64)} chars")

        # Query all models in parallel
        print(f"🔄 Querying {len(self.models)} models in parallel...")

        tasks = []
        for model in self.models:
            task = self._query_single_local_model(model, image_base64, text_base64, ocr_text, analysis_mode, custom_prompt)
            tasks.append(task)

        # Wait for all responses
        model_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful_results = []
        all_responses = {}

        for i, (model, result) in enumerate(zip(self.models, model_results)):
            model_name = model["name"]
            print(f"\n{i+1}. {model_name}:")

            if isinstance(result, Exception):
                print(f"   ❌ Exception: {result}")
                all_responses[model_name] = {
                    "status": "exception",
                    "error": str(result)
                }
            else:
                status = result.get("status", "unknown")
                raw_resp = result.get("raw_response", "No response")
                print(f"   📊 Status: {status}")
                print(f"   📤 Response: '{raw_resp[:100]}...' ({len(raw_resp)} chars)")

                all_responses[model_name] = result

                # Check if we got valid parsed data
                if result.get("parsed_data"):
                    parsed_data = result["parsed_data"]

                    # Validate based on analysis mode
                    is_valid = False
                    if analysis_mode == "ui":
                        # UI mode: look for categories OR hierarchy fields
                        if parsed_data.get("categories") and parsed_data["categories"] != ['']:
                            print(f"   ✅ Categories: {parsed_data['categories']}")
                            is_valid = True
                        elif parsed_data.get("main_category") or parsed_data.get("active_subcategory"):
                            print(f"   ✅ Hierarchy: main='{parsed_data.get('main_category')}', active='{parsed_data.get('active_subcategory')}'")
                            is_valid = True
                    elif analysis_mode == "coordinate_mapping":
                        # Coordinate mapping mode: look for category region analysis
                        if parsed_data.get("category_region_analysis"):
                            print(f"   ✅ Coordinate mapping data found")
                            is_valid = True
                    else:
                        # Product mode: look for product info
                        if parsed_data.get("product_name") or parsed_data.get("price"):
                            print(f"   ✅ Product: {parsed_data}")
                            is_valid = True

                    if is_valid:
                        successful_results.append({
                            "model": model_name,
                            "data": parsed_data,
                            "weight": model["weight"],
                            "status": status
                        })
                    else:
                        print(f"   ⚠️  Invalid data for {analysis_mode} mode")
                else:
                    print(f"   ❌ No valid parsed data")

        # Create consensus result based on mode
        if successful_results:
            if analysis_mode == "ui":
                # UI mode: return categories with PROPER MAJORITY VOTING
                all_categories = []
                available_subcategories = []
                visual_hierarchy = None

                # Collect votes for main_category and active_subcategory
                main_category_votes = {}
                active_subcategory_votes = {}

                for result in successful_results:
                    result_data = result["data"]

                    # Vote counting for main_category
                    if result_data.get("main_category"):
                        main_cat = result_data.get("main_category").strip()
                        if main_cat and main_cat not in ["text from pink background row", ""]:  # Filter out template responses
                            main_category_votes[main_cat] = main_category_votes.get(main_cat, 0) + 1

                    # Vote counting for active_subcategory
                    if result_data.get("active_subcategory"):
                        active_sub = result_data.get("active_subcategory").strip()
                        if active_sub and active_sub not in ["bold/centered text from bottom row", ""]:  # Filter out template responses
                            active_subcategory_votes[active_sub] = active_subcategory_votes.get(active_sub, 0) + 1

                    # Collect available subcategories and other data
                    if result_data.get("available_subcategories"):
                        available_subcategories.extend(result_data.get("available_subcategories", []))
                    if result_data.get("visual_hierarchy"):
                        visual_hierarchy = result_data.get("visual_hierarchy")

                    # Also collect categories field for backwards compatibility
                    categories = result_data.get("categories", [])
                    normalized_categories = []
                    for cat in categories:
                        if isinstance(cat, dict):
                            normalized_categories.append(cat.get("name", str(cat)))
                        else:
                            normalized_categories.append(str(cat))
                    all_categories.extend(normalized_categories)

                # MAJORITY VOTING: Pick the answer with most votes
                main_category = max(main_category_votes, key=main_category_votes.get) if main_category_votes else None
                active_subcategory = max(active_subcategory_votes, key=active_subcategory_votes.get) if active_subcategory_votes else None

                # Debug output for vote counts
                print(f"   🗳️ Main category votes: {main_category_votes}")
                print(f"   🗳️ Active subcategory votes: {active_subcategory_votes}")
                if main_category:
                    print(f"   🏆 Majority winner main: '{main_category}' ({main_category_votes.get(main_category, 0)} votes)")
                if active_subcategory:
                    print(f"   🏆 Majority winner active: '{active_subcategory}' ({active_subcategory_votes.get(active_subcategory, 0)} votes)")

                unique_categories = list(set(all_categories))
                unique_available_subcategories = list(set(available_subcategories)) if available_subcategories else []

                consensus_result = {
                    "categories": unique_categories,
                    "main_category": main_category,
                    "active_subcategory": active_subcategory,
                    "available_subcategories": unique_available_subcategories,
                    "visual_hierarchy": visual_hierarchy,
                    "successful_models": len(successful_results),
                    "total_models": len(self.models),
                    "confidence": len(successful_results) / len(self.models),
                    "individual_results": successful_results,
                    "analysis_method": "consensus",
                    "analysis_mode": analysis_mode,
                    "ocr_text": ocr_text
                }

                print(f"\n🎯 UI CONSENSUS SUCCESS:")
                print(f"   ✅ Found {len(unique_categories)} categories: {unique_categories}")
                if main_category:
                    print(f"   🏗️ Main category: '{main_category}'")
                if active_subcategory:
                    print(f"   🎯 Active subcategory: '{active_subcategory}'")
                if unique_available_subcategories:
                    print(f"   📋 Available subcategories: {unique_available_subcategories}")
                print(f"   📊 {len(successful_results)}/{len(self.models)} models succeeded")

            elif analysis_mode == "coordinate_mapping":
                # Coordinate mapping mode: return the best coordinate analysis
                best_result = successful_results[0]  # Take first successful result
                coordinate_data = best_result["data"]
                consensus_result = coordinate_data  # Return the coordinate analysis directly

                print(f"\n🎯 COORDINATE MAPPING SUCCESS:")
                print(f"   ✅ Found coordinate guidance from LLM")
                print(f"   📊 {len(successful_results)}/{len(self.models)} models succeeded")

            else:
                # Product mode: return weighted consensus of product data
                # Use the highest weighted successful result
                best_result = max(successful_results, key=lambda x: x["weight"])
                product_data = best_result["data"]

                # Calculate cost metrics from the extracted data
                cost_metrics = self._calculate_cost_metrics(product_data)

                consensus_result = {
                    "price": product_data.get("price", ""),
                    "brand": product_data.get("brand", ""),
                    "product_name": product_data.get("product_name", ""),
                    "weight": product_data.get("weight", ""),
                    "quantity": product_data.get("quantity", ""),
                    "unit": product_data.get("unit", ""),
                    "cost_per_kg": cost_metrics.get("cost_per_kg", ""),
                    "cost_per_piece": cost_metrics.get("cost_per_piece", ""),
                    "successful_models": len(successful_results),
                    "total_models": len(self.models),
                    "confidence": len(successful_results) / len(self.models),
                    "individual_results": successful_results,
                    "analysis_method": "consensus",
                    "analysis_mode": analysis_mode,
                    "ocr_text": ocr_text
                }

                print(f"\n🎯 PRODUCT CONSENSUS SUCCESS:")
                print(f"   ✅ Best result: {product_data}")
                if cost_metrics.get("cost_per_kg"):
                    print(f"   💰 Cost per kg: {cost_metrics['cost_per_kg']}")
                if cost_metrics.get("cost_per_piece"):
                    print(f"   💰 Cost per piece: {cost_metrics['cost_per_piece']}")
                print(f"   📊 {len(successful_results)}/{len(self.models)} models succeeded")

        else:
            print(f"\n❌ CONSENSUS FAILED:")
            print(f"   📊 0/{len(self.models)} models provided valid {analysis_mode} data")

            if analysis_mode == "ui":
                # Return empty categories structure
                consensus_result = {
                    "categories": [],
                    "successful_models": 0,
                    "total_models": len(self.models),
                    "confidence": 0.0,
                    "individual_results": [],
                    "analysis_method": "failed_consensus",
                    "analysis_mode": analysis_mode,
                    "ocr_text": ocr_text
                }
            else:
                # Return empty product structure
                consensus_result = {
                    "price": "",
                    "brand": "",
                    "product_name": "",
                    "weight": "",
                    "quantity": "",
                    "unit": "",
                    "cost_per_kg": "",
                    "cost_per_piece": "",
                    "successful_models": 0,
                    "total_models": len(self.models),
                    "confidence": 0.0,
                    "individual_results": [],
                    "analysis_method": "failed_consensus",
                    "analysis_mode": analysis_mode,
                    "ocr_text": ocr_text
                }

        # Add debug info
        consensus_result["debug"] = {
            "all_model_responses": all_responses,
            "prompt_mode": analysis_mode
        }

        return consensus_result

    async def analyze_header_coordinates(self, header_image: np.ndarray) -> Dict:
        """
        Analyze header image to determine optimal coordinates for category extraction.
        Uses LLM consensus to identify where category information is located.
        """
        coordinate_prompt = """
You are analyzing a Flink grocery app header screenshot. Your job is to identify WHERE the category hierarchy is visually displayed.

VISUAL HIERARCHY TO IDENTIFY:
1. **MAIN CATEGORY** = Text with colored/pink background or highlighted appearance
2. **ACTIVE SUBCATEGORY** = Bold text underneath the main category
3. **INACTIVE SUBCATEGORIES** = Regular text alongside the active subcategory
4. **UNSELECTED CATEGORIES** = Regular text on sides (like "Käse", "Fleisch", etc.)

TASK: Find the precise region containing this category hierarchy and provide crop coordinates.

Look specifically for:
- PINK/COLORED BACKGROUNDS indicating selected main category
- BOLD TEXT indicating active subcategory
- TEXT POSITIONING showing category → subcategory relationship
- The complete category navigation area (not just individual elements)

Respond in this EXACT JSON format:

{
  "category_region_analysis": {
    "text_dense_area_description": "describe the visual layout of categories and subcategories",
    "main_category_visual": {
      "name": "the category with pink/colored background",
      "styling": "describe its visual appearance (background color, highlighting)"
    },
    "subcategories_visual": {
      "active_subcategory": "the bold text subcategory name",
      "inactive_subcategories": ["list", "of", "regular", "text", "subcategories"]
    },
    "unselected_categories": ["visible", "side", "categories"],
    "optimal_crop_region": {
      "method": "smart_category_focus",
      "start_percentage": 50,
      "end_percentage": 100,
      "reasoning": "captures full category hierarchy including main category and subcategories"
    },
    "visual_hierarchy_location": {
      "main_category_position": "describe where the highlighted main category appears",
      "subcategory_position": "describe where subcategories appear relative to main category",
      "complete_navigation_area": "describe the full category navigation region boundaries"
    }
  }
}

CRITICAL: Focus on the VISUAL STYLING (pink backgrounds, bold text) that indicates category selection state. The goal is to capture the complete category navigation hierarchy, not just individual text elements.
"""

        print("🗺️ COORDINATE MAPPING: Analyzing header for optimal category region...")

        # Use the main consensus method with coordinate-specific prompt
        result = await self.analyze_product_with_consensus(
            header_image,
            header_image,
            "coordinate_mapping",
            custom_prompt=coordinate_prompt
        )

        return result

    # Legacy method for backward compatibility
    async def analyze_categories_with_consensus(self, image: np.ndarray) -> Dict:
        """Legacy method - redirects to main analyze_product_with_consensus method."""
        return await self.analyze_product_with_consensus(image, image, "ui")