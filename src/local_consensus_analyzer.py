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

    def _create_local_analysis_prompt(self, ocr_text: str, analysis_mode: str) -> str:
        """Create analysis prompt based on mode (ui/product)."""
        if analysis_mode == "ui":
            # UI analysis for category extraction
            return f"""Look at this Flink grocery app navigation screenshot. Find ALL visible food category names in German.

OCR detected text: "{ocr_text}"

Extract ACTUAL visible category names. Look for German food categories like:
- Obst, Gemüse, Bananen, Backwaren, Fleisch & Fisch, Joghurt & Desserts, Äpfel & Birnen, Getränke
- Vegan & Vegetarisch, Milchalternativen, Bio, Tiefkühl, Nudeln
- Any other visible German category names in the navigation

Find the currently highlighted/active category if visible.

Return JSON with REAL category names from the image:
{{"categories": ["actual_category_1", "actual_category_2", "..."], "current": "currently_active_category_name"}}

Extract the actual German category names visible in the image. Do NOT use placeholder names. Return only JSON."""

        else:
            # Product analysis for product extraction
            return f"""Analyze this grocery product image. Extract complete product information and return JSON.

OCR detected text: "{ocr_text}"

Extract ALL visible information:
- Product name (German name as shown)
- Price (in euros €, including decimals like 1,49 €)
- Brand name (if visible on product or package)
- Weight/quantity (kg, g, ml, l, Stk. - look for numbers followed by units)
- Quantity count (for items sold by pieces like "5 Stk.")

Examples of expected output:
- For bananas sold by piece: {{"price": "1,49 €", "product_name": "Bananen", "brand": "Chiquita", "weight": "", "quantity": "5", "unit": "Stk"}}
- For apples sold by weight: {{"price": "3,99 €", "product_name": "Äpfel Pink Lady", "brand": "Pink Lady", "weight": "1kg", "quantity": "", "unit": "kg"}}

Be thorough - look at both the main product image and any text areas for brand names, measurements, and quantities.

Return only JSON format:
{{"price": "", "product_name": "", "brand": "", "weight": "", "quantity": "", "unit": ""}}

Extract information from both the image and OCR text. Return only the JSON, no other text."""

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

    async def _query_single_local_model(self, model: Dict, image_base64: str, text_base64: str, ocr_text: str, analysis_mode: str) -> Dict:
        """Query single Ollama model with proper error handling."""
        model_name = model["name"]
        start_time = time.time()

        try:
            # Create appropriate prompt
            prompt = self._create_local_analysis_prompt(ocr_text, analysis_mode)

            # Prepare payload for vision models
            payload = {
                "model": model_name,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
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
                        # Extract JSON from text
                        json_match = re.search(r'\{[^}]*\}', raw_content)
                        if json_match:
                            json_str = json_match.group()
                            parsed_json = json.loads(json_str)
                            return {
                                "status": "extracted_json",
                                "raw_response": raw_content,
                                "parsed_data": parsed_json,
                                "processing_time": processing_time,
                                "parse_status": "extracted_json"
                            }
                        else:
                            return {
                                "status": "no_json",
                                "raw_response": raw_content,
                                "error": "No JSON found in response",
                                "processing_time": processing_time,
                                "parse_status": "no_json"
                            }

                except json.JSONDecodeError as e:
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
            return {
                "status": "exception",
                "raw_response": "Exception occurred",
                "error": str(e),
                "processing_time": processing_time
            }

    async def analyze_product_with_consensus(self, tile_image: np.ndarray, text_region_image: np.ndarray, analysis_mode: str = "product") -> Dict:
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
        text_base64 = image_base64  # Using same image for both

        print(f"🖼️  Image encoded: {len(image_base64)} chars")

        # Query all models in parallel
        print(f"🔄 Querying {len(self.models)} models in parallel...")

        tasks = []
        for model in self.models:
            task = self._query_single_local_model(model, image_base64, text_base64, ocr_text, analysis_mode)
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
                        # UI mode: look for categories
                        if parsed_data.get("categories") and parsed_data["categories"] != ['']:
                            print(f"   ✅ Categories: {parsed_data['categories']}")
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
                # UI mode: return categories
                all_categories = []
                for result in successful_results:
                    categories = result["data"].get("categories", [])
                    # Normalize categories - handle both strings and dictionaries
                    normalized_categories = []
                    for cat in categories:
                        if isinstance(cat, dict):
                            # Extract name from dictionary format
                            normalized_categories.append(cat.get("name", str(cat)))
                        else:
                            # Already a string
                            normalized_categories.append(str(cat))
                    all_categories.extend(normalized_categories)

                unique_categories = list(set(all_categories))

                consensus_result = {
                    "categories": unique_categories,
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

    # Legacy method for backward compatibility
    async def analyze_categories_with_consensus(self, image: np.ndarray) -> Dict:
        """Legacy method - redirects to main analyze_product_with_consensus method."""
        return await self.analyze_product_with_consensus(image, image, "ui")